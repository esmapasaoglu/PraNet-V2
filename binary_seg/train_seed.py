"""Multi-seed training script for PraNet-V1 / PVT-PraNet-V1 / PraNet-V2 /
PVT-PraNet-V2 and the background-supervision ablation.

It reproduces the official training loop of MyTrain_med.py (Adam, lr 1e-4,
batch 8, 352x352, multi-scale {0.75, 1, 1.25}, gradient clamp 0.5, 29 epochs,
constant learning rate) and adds what the study needs:

  * --seed            : seeds weight init (global torch RNG) and, through a
                        separate generator, the data order
  * V1 models         : the official repo only trains V2; V1 uses the
                        original PraNet loss (wBCE + wIoU) with the same recipe
  * --bg_weight       : 0.8 = official DSRA loss, 0 = our ablation
  * last-epoch weights: the official script only saves every 10 epochs and the
                        test-selected best.pth; our protocol evaluates the
                        final epoch, saved as last.pth
  * resumable checkpoints for interrupted Colab/Kaggle sessions
  * a JSON log with GPU type, versions, epoch times and losses

No test set is touched during training (the official script evaluates
CVC-300 / CVC-ClinicDB every epoch and keeps the best one; we do not).

Usage (from the binary_seg directory):
    python train_seed.py --model_type PraNet-V2 --seed 1 --out_dir runs/PraNet-V2_s1
"""
import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from proje_common import (MODEL_TYPES, build_model, device_name, get_device, is_v2, seed_worker,
                          set_seed, total_loss)
# Importing the official dataloader module sets torch.backends.cudnn.enabled =
# False as a side effect (utils/dataloader.py:21). We re-apply our own choice below.
from utils.dataloader import PolypDataset

SIZE_RATES = [0.75, 1, 1.25]  # official multi-scale training (MyTrain_med.py:55)


def clip_gradient(optimizer, grad_clip):
    """Same as utils.utils.clip_gradient (copied to avoid the thop import)."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate, decay_epoch):
    """Same as utils.utils.adjust_lr. With decay_epoch=50 and 29 epochs the
    factor is always 1, i.e. the learning rate stays constant."""
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def rng_state(generator):
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'data_generator': generator.get_state(),
    }


def set_rng_state(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['data_generator'])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_type', required=True, choices=MODEL_TYPES)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out_dir', required=True, help='where checkpoints and logs go (e.g. on Google Drive)')
    p.add_argument('--bg_weight', type=float, default=0.8,
                   help='weight of the background BCE term for V2 models; 0 = ablation')
    p.add_argument('--train_path', default='./data/TrainDataset')
    # Official recipe (MyTrain_med.py defaults). epoch=30 with range(1, 30) -> 29 epochs.
    p.add_argument('--epochs', type=int, default=29)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batchsize', type=int, default=8)
    p.add_argument('--accum_steps', type=int, default=1,
                   help='gradient accumulation; batchsize*accum_steps must equal 8 to keep the recipe')
    p.add_argument('--trainsize', type=int, default=352)
    p.add_argument('--clip', type=float, default=0.5)
    p.add_argument('--decay_rate', type=float, default=0.1)
    p.add_argument('--decay_epoch', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=2)
    # Implementation switches (not part of the recipe)
    p.add_argument('--cudnn', type=int, default=0,
                   help='official code disables cuDNN (0). 1 enables it (faster); decide in the pilot')
    p.add_argument('--deterministic', type=int, default=1)
    p.add_argument('--save_every', type=int, default=10, help='also keep epoch_N.pth every N epochs (0 = off)')
    p.add_argument('--max_epochs_this_session', type=int, default=0,
                   help='stop cleanly after N epochs (to split a run over sessions); 0 = no limit')
    p.add_argument('--smoke', type=int, default=0,
                   help='>0: run only this many batches per epoch (pipeline test, results meaningless)')
    p.add_argument('--log_first_steps', type=int, default=10,
                   help='record the loss of every optimizer step of the first N batches (determinism check)')
    return p.parse_args()


def main():
    opt = parse_args()
    if opt.batchsize * opt.accum_steps != 8:
        print('WARNING: effective batch size {} differs from the official 8'.format(
            opt.batchsize * opt.accum_steps))
    if not is_v2(opt.model_type) and opt.bg_weight != 0.8:
        print('NOTE: --bg_weight is ignored for V1 models (they have no background branch)')

    os.makedirs(opt.out_dir, exist_ok=True)
    ckpt_path = os.path.join(opt.out_dir, 'ckpt_latest.pth')
    log_path = os.path.join(opt.out_dir, 'train_log.json')
    if os.path.exists(os.path.join(opt.out_dir, 'last.pth')):
        print('Run already finished: {}/last.pth exists. Nothing to do.'.format(opt.out_dir))
        return

    set_seed(opt.seed, deterministic=bool(opt.deterministic))
    torch.backends.cudnn.enabled = bool(opt.cudnn)

    # Separate RNG for the data order, so the shuffling sequence depends only on
    # the seed and not on how many random numbers model construction consumed.
    data_gen = torch.Generator()
    data_gen.manual_seed(opt.seed)

    image_root = '{}/images/'.format(opt.train_path)
    if not os.path.isdir(image_root) and os.path.isdir('{}/image/'.format(opt.train_path)):
        # The official TrainDataset.zip names the folder 'image' (singular), while
        # MyTrain_med.py expects 'images'; accept both.
        image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    dataset = PolypDataset(image_root, gt_root, opt.trainsize)
    device = get_device()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers,
        pin_memory=(device.type == 'cuda'), worker_init_fn=seed_worker, generator=data_gen)
    total_step = len(loader)

    model = build_model(opt.model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 1
    log = {
        'args': vars(opt),
        'gpu': device_name(device),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'python': platform.python_version(),
        'train_images': len(dataset),
        'n_params': sum(p.numel() for p in model.parameters()),
        'started': datetime.now().isoformat(timespec='seconds'),
        'sessions': [],
        'epochs': [],
        'first_steps': [],  # [epoch, batch, rate, loss] for the first --log_first_steps batches
    }
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        set_rng_state(ckpt['rng'], data_gen)
        start_epoch = ckpt['epoch'] + 1
        if os.path.exists(log_path):
            with open(log_path) as f:
                log = json.load(f)
        print('Resumed from epoch {}'.format(ckpt['epoch']))
    log['sessions'].append({'start_epoch': start_epoch, 'gpu': device_name(device),
                            'time': datetime.now().isoformat(timespec='seconds')})

    print('{} | seed {} | bg_weight {} | {} images | {} steps/epoch | GPU {}'.format(
        opt.model_type, opt.seed, opt.bg_weight, len(dataset), total_step, log['gpu']))

    epochs_this_session = 0
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_lr(optimizer, epoch, opt.decay_rate, opt.decay_epoch)
        model.train()
        t0 = time.time()
        loss_sum, loss_n = 0.0, 0
        for i, (images, gts) in enumerate(loader, start=1):
            if opt.smoke and i > opt.smoke:
                break
            for rate in SIZE_RATES:
                optimizer.zero_grad()
                # Gradient accumulation only matters if the batch must be split for memory.
                img_chunks = images.chunk(opt.accum_steps)
                gt_chunks = gts.chunk(opt.accum_steps)
                rate_loss = 0.0
                for im, gt in zip(img_chunks, gt_chunks):
                    im = im.to(device, non_blocking=True)
                    gt = gt.to(device, non_blocking=True)
                    trainsize = int(round(opt.trainsize * rate / 32) * 32)
                    if rate != 1:
                        im = F.interpolate(im, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        gt = F.interpolate(gt, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    outputs = model(im)
                    loss = total_loss(opt.model_type, outputs, gt, opt.bg_weight)
                    # Each chunk's loss is a mean over its own samples; weight by chunk size
                    # so the accumulated gradient equals that of the full batch.
                    (loss * im.shape[0] / images.shape[0]).backward()
                    rate_loss += loss.item() * im.shape[0] / images.shape[0]
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                if epoch == 1 and i <= opt.log_first_steps:
                    log['first_steps'].append([epoch, i, rate, rate_loss])
                if rate == 1:
                    loss_sum += rate_loss
                    loss_n += 1
            if i % 20 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}] Step [{:04d}/{:04d}] loss(rate=1) {:.4f}'.format(
                    datetime.now().strftime('%H:%M:%S'), epoch, opt.epochs, i, total_step,
                    loss_sum / max(loss_n, 1)))
                sys.stdout.flush()

        epoch_time = time.time() - t0
        log['epochs'].append({'epoch': epoch, 'seconds': round(epoch_time, 1),
                              'loss': loss_sum / max(loss_n, 1)})
        if opt.save_every and epoch % opt.save_every == 0:
            torch.save(model.state_dict(), os.path.join(opt.out_dir, 'epoch_{}.pth'.format(epoch)))
        # Resumable checkpoint (overwritten every epoch). Write to a temp file
        # first so an interrupted save cannot corrupt the previous checkpoint.
        tmp = ckpt_path + '.tmp'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'rng': rng_state(data_gen)}, tmp)
        os.replace(tmp, ckpt_path)
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=1)
        print('Epoch {} done in {:.0f}s'.format(epoch, epoch_time))

        epochs_this_session += 1
        if opt.max_epochs_this_session and epochs_this_session >= opt.max_epochs_this_session \
                and epoch < opt.epochs:
            print('Stopping after {} epochs this session; rerun the same command to resume.'.format(
                epochs_this_session))
            return

    torch.save(model.state_dict(), os.path.join(opt.out_dir, 'last.pth'))
    log['finished'] = datetime.now().isoformat(timespec='seconds')
    log['total_train_seconds'] = round(sum(e['seconds'] for e in log['epochs']), 1)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=1)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # the optimizer state is no longer needed
    print('Finished. Final weights: {}'.format(os.path.join(opt.out_dir, 'last.pth')))


if __name__ == '__main__':
    main()
