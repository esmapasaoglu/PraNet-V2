"""Numerical check of the code-reading finding about DSRA in binary mode.

lib/pranet.py fuses the previous stage as
    ra_feat_fg = ra_feat_fg + ra_feat_fg * softmax(crop_fg - crop_bg, dim=1)
With num_class=1 (all polyp models) the softmax runs over a single channel and
is identically 1, so the previous stage's prediction should have NO effect on
the later side outputs.

Test: perturb the coarse decoder output (agg1 -> map5 fg/bg) with large noise
through a forward hook and measure how much map2/map3/map4 change.
  * expected for use_softmax=True  (official): change ~ 0, map5 changes
  * control  for use_softmax=False            : map2/3/4 DO change
The control shows that the hook really reaches the later stages.

Weights are the ImageNet backbones + random decoder (the fusion path is
independent of the trained values). Pass --ckpt to use trained weights instead.

Usage (from binary_seg):  python check_dsra_noop.py --model_type PraNet-V2
"""
import argparse

import torch

from proje_common import get_device, load_weights


def build(model_type, use_softmax):
    from lib.pranet import PraNet_V2, PVT_PraNet_V2
    cls = PraNet_V2 if model_type == 'PraNet-V2' else PVT_PraNet_V2
    return cls(num_class=1, use_softmax=use_softmax)


def max_changes(model, x):
    def hook(module, inputs, output):
        fg, bg = output
        return fg + 10 * torch.randn_like(fg), bg + 10 * torch.randn_like(bg)

    with torch.no_grad():
        base = model(x)
        base_again = model(x)
        handle = model.agg1.register_forward_hook(hook)
        pert = model(x)
        handle.remove()
    names = ['map2_fg', 'map3_fg', 'map4_fg', 'map5_fg']
    noise_floor = max((a - b).abs().max().item() for a, b in zip(base[:4], base_again[:4]))
    changes = {n: (b - p).abs().max().item() for n, b, p in zip(names, base[:4], pert[:4])}
    return noise_floor, changes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_type', default='PraNet-V2', choices=['PraNet-V2', 'PVT-PraNet-V2'])
    p.add_argument('--ckpt', default=None)
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(0)
    x = torch.randn(2, 3, 352, 352).to(device)
    verdict = {}
    for use_softmax in (True, False):
        torch.manual_seed(0)
        model = build(args.model_type, use_softmax)
        if args.ckpt:
            load_weights(model, args.ckpt)
        model.to(device).eval()
        floor, changes = max_changes(model, x)
        label = 'use_softmax={} ({})'.format(use_softmax, 'official' if use_softmax else 'control')
        print(label)
        print('   repeat-run noise floor: {:.2e}'.format(floor))
        for k, v in changes.items():
            print('   max |change| {:<8}: {:.4e}'.format(k, v))
        verdict[use_softmax] = max(changes['map2_fg'], changes['map3_fg'], changes['map4_fg'])

    tol = 1e-4
    print()
    if verdict[True] <= tol and verdict[False] > tol:
        print('CONFIRMED: with the official setting, the previous-stage prediction has no effect '
              'on map2-map4 (fusion is a no-op for num_class=1); the control shows the path exists.')
    elif verdict[True] > tol:
        print('NOT CONFIRMED: the previous stage does influence map2-map4. Re-check the code reading.')
    else:
        print('INCONCLUSIVE: the control did not change either; the hook may not reach the later stages.')


if __name__ == '__main__':
    main()
