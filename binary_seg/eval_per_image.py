"""Evaluate one checkpoint on the polyp test sets and save PER-IMAGE metrics.

For every test image it stores meanDic / meanIoU (eval.py definitions: mean
over 256 thresholds) under BOTH inference protocols found in MyTest_med.py:

    last : only the final side output (official protocol for V1)
    sum  : sum of the four side outputs (official protocol for V2)

so the protocol confound can be analysed without extra GPU time. With --full
the remaining eval.py metrics (Sm, wFm, meanEm, mae) are added for the model's
official protocol only (they are slow).

The image mean of meanDic equals the dataset mDice printed by eval.py.

Usage (from binary_seg):
    python eval_per_image.py --model_type PraNet-V2 --ckpt runs/PraNet-V2_s1/last.pth \
        --out_dir results_per_image/PraNet-V2_s1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from proje_common import (MODEL_TYPES, TEST_SETS, build_model, device_name, get_device, image_metrics,
                          load_weights, official_protocol, predict_uint8, protocol_logits, split_outputs)
from utils.dataloader import test_dataset  # note: sets cudnn.enabled = False, as in the official test


def read_gt_raw(path):
    """Read the mask exactly like eval.py (raw values, first channel if RGB)."""
    gt = np.array(Image.open(path))
    if gt.ndim != 2:
        gt = gt[:, :, 0]
    return gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_type', required=True, choices=MODEL_TYPES)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--data_root', default='./data/TestDataset')
    p.add_argument('--datasets', nargs='+', default=TEST_SETS)
    p.add_argument('--testsize', type=int, default=352)
    p.add_argument('--full', type=int, default=0, help='1 = also Sm, wFm, meanEm, mae (slow)')
    p.add_argument('--save_png', type=int, default=0, help='1 = also save official-protocol maps as PNG')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = build_model(args.model_type)
    load_report = load_weights(model, args.ckpt)
    if load_report['missing_keys'] or load_report['unexpected_keys']:
        print('Key report:', json.dumps(load_report)[:500])
    device = get_device()
    model.to(device).eval()
    official = official_protocol(args.model_type)

    summary = {'model_type': args.model_type, 'ckpt': os.path.abspath(args.ckpt),
               'official_protocol': official, 'load_report': load_report,
               'gpu': device_name(device), 'datasets': {}}

    for ds in args.datasets:
        img_root = os.path.join(args.data_root, ds, 'images') + '/'
        gt_root = os.path.join(args.data_root, ds, 'masks') + '/'
        if not os.path.isdir(img_root):
            print('!! Missing test set {}, skipped'.format(ds))
            summary['datasets'][ds] = 'missing'
            continue
        loader = test_dataset(img_root, gt_root, args.testsize)
        assert len(loader.images) == len(loader.gts), 'image/mask count mismatch in ' + ds
        if args.save_png:
            png_dir = os.path.join(args.out_dir, 'png', ds)
            os.makedirs(png_dir, exist_ok=True)

        rows = []
        t0 = time.time()
        for _ in range(loader.size):
            gt_path = loader.gts[loader.index]
            image, _gt_l, name = loader.load_data()
            assert os.path.splitext(os.path.basename(gt_path))[0] == os.path.splitext(name)[0], \
                'image/mask name mismatch: {} vs {}'.format(name, gt_path)
            gt_raw = read_gt_raw(gt_path)

            with torch.no_grad():
                outputs = model(image.to(device))
            fg, _ = split_outputs(args.model_type, outputs)

            row = {'image': name, 'polyp_fraction': float((gt_raw > 0.5).mean())}
            for proto, logits in protocol_logits(fg).items():
                pred = predict_uint8(logits, gt_raw.shape)
                m = image_metrics(pred, gt_raw, full=bool(args.full) and proto == official)
                for k, v in m.items():
                    row['{}_{}'.format(k, proto)] = v
                if args.save_png and proto == official:
                    Image.fromarray(pred).save(os.path.join(png_dir, name))
            rows.append(row)

        cols = list(rows[0].keys())
        with open(os.path.join(args.out_dir, '{}.csv'.format(ds)), 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in rows:
                f.write(','.join(str(r[c]) for c in cols) + '\n')
        means = {c: float(np.mean([r[c] for r in rows])) for c in cols if c not in ('image',)}
        means['n_images'] = len(rows)
        summary['datasets'][ds] = means
        print('{:<20} n={:<4} mDice(official={}) {:.4f} | last {:.4f} | sum {:.4f}  ({:.0f}s)'.format(
            ds, len(rows), official, means['meanDic_' + official], means['meanDic_last'],
            means['meanDic_sum'], time.time() - t0))

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
