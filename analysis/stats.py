"""Pre-registered statistical analysis of the multi-seed study.

Implements the decision rule of the proposal (section 2):
  * two 95% CIs per difference: hierarchical bootstrap (seeds resampled
    independently per model, test images resampled with ONE shared index set
    per bootstrap draw, i.e. images are paired across models) and a Welch t
    interval on the seed means (Welch-Satterthwaite df for contrasts);
  * SESOI = 1 mDice point; four categories assigned top-down:
        1 consistent & practically important: both lower bounds > 0 and mean >= 1
        2 consistent but small              : both lower bounds > 0, mean < 1
        3 practically absent or negative    : both upper bounds < 1
        4 inconclusive                      : everything else
  * S4: 95% prediction interval  xbar +- t_{0.975,n-1} * s * sqrt(1 + 1/n)
  * S5: retained-gain ratio rho = (Abl - V1) / (V2 - V1) with a bootstrap CI,
        only computed if the PVT V2-V1 gain is in category 1 or 2;
        H5 supported if CI_upper(rho) < 0.5 AND Welch lower bound of (V2 - Abl) > 0,
        rejected if CI_lower(rho) > 0.5, otherwise inconclusive.
  * variance decomposition: bootstrap variance of a model mean when only seeds
    vs. only images are resampled.

All numbers are in mDice points (x100).

Input layout (written by eval_per_image.py):
    <results_root>/<config>_s<seed>/<dataset>.csv   with columns image, meanDic_last, meanDic_sum, ...
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

SESOI = 1.0
CATEGORY_NAMES = {1: 'tutarlı ve önemli', 2: 'tutarlı ama küçük', 3: 'pratik olarak yok/negatif',
                  4: 'belirsiz'}
OFFICIAL = {'PraNet-V1': 'last', 'PVT-PraNet-V1': 'last', 'PraNet-V2': 'sum', 'PVT-PraNet-V2': 'sum',
            'PVT-PraNet-V2-abl': 'sum'}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_runs(results_root, dataset, protocol='official', metric='meanDic'):
    """Return {config: matrix [n_seeds, n_images]} (x100) and the image order.

    protocol: 'official' (each model's own protocol), 'last' or 'sum'.
    Images are aligned by name across all runs.
    """
    per_config = {}
    for d in sorted(glob.glob(os.path.join(results_root, '*_s*'))):
        m = re.match(r'(.+)_s(\d+)$', os.path.basename(d))
        csv = os.path.join(d, dataset + '.csv')
        if not m or not os.path.exists(csv):
            continue
        config, seed = m.group(1), int(m.group(2))
        proto = OFFICIAL[config] if protocol == 'official' else protocol
        df = pd.read_csv(csv).set_index('image')
        per_config.setdefault(config, {})[seed] = 100 * df['{}_{}'.format(metric, proto)]
    images = None
    for seeds in per_config.values():
        for s in seeds.values():
            images = s.index if images is None else images.intersection(s.index)
    images = sorted(images) if images is not None else []
    out = {c: np.vstack([seeds[k].loc[images].to_numpy() for k in sorted(seeds)])
           for c, seeds in per_config.items()}
    return out, images


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #
def welch_contrast_ci(groups, weights, alpha=0.05):
    """CI of sum_k w_k * mean(seed means of group k) with Welch-Satterthwaite df.

    groups: list of 1-D arrays of seed-level means. Two groups with weights
    (+1, -1) give the classical Welch interval; four groups give the
    interaction contrast.
    """
    est = sum(w * g.mean() for g, w in zip(groups, weights))
    terms = [w ** 2 * g.var(ddof=1) / len(g) for g, w in zip(groups, weights)]
    se = np.sqrt(sum(terms))
    if se == 0:
        return est, est, est
    df = se ** 4 / sum(t ** 2 / (len(g) - 1) for t, g in zip(terms, groups))
    h = stats.t.ppf(1 - alpha / 2, df) * se
    return est, est - h, est + h


def hierarchical_bootstrap(mats, weights, n_boot=10000, seed=0, mode='both', alpha=0.05,
                           ratio=None):
    """Bootstrap distribution of a linear contrast (or a ratio) of model means.

    mats: list of [n_seeds_k, n_images] arrays (same image order).
    mode: 'both' (seeds and images), 'seeds' (only seeds) or 'images' (only images).
    ratio: optional (num_weights, den_weights) -> returns the distribution of
           contrast(num) / contrast(den) instead of contrast(weights).
    Returns (draws, lo, hi).
    """
    rng = np.random.default_rng(seed)
    n_img = mats[0].shape[1]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_img, n_img) if mode in ('both', 'images') else slice(None)
        means = []
        for m in mats:
            rows = rng.integers(0, m.shape[0], m.shape[0]) if mode in ('both', 'seeds') \
                else np.arange(m.shape[0])
            means.append(m[rows][:, idx].mean())
        means = np.array(means)
        if ratio is None:
            draws[b] = means @ np.asarray(weights, float)
        else:
            num, den = (means @ np.asarray(w, float) for w in ratio)
            draws[b] = num / den if den != 0 else np.nan
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return draws, lo, hi


def categorize(mean, ci_boot, ci_welch, sesoi=SESOI):
    lows = (ci_boot[0], ci_welch[0])
    highs = (ci_boot[1], ci_welch[1])
    if min(lows) > 0 and mean >= sesoi:
        return 1
    if min(lows) > 0:
        return 2
    if max(highs) < sesoi:
        return 3
    return 4


def prediction_interval(x, alpha=0.05):
    x = np.asarray(x, float)
    n = len(x)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) * np.sqrt(1 + 1 / n)
    return x.mean() - h, x.mean() + h


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #
def compare(mats, groups, weights, n_boot=10000):
    """Full decision-rule output for one contrast."""
    sel = [mats[g] for g in groups]
    seed_means = [m.mean(axis=1) for m in sel]
    est, wlo, whi = welch_contrast_ci(seed_means, weights)
    _, blo, bhi = hierarchical_bootstrap(sel, weights, n_boot=n_boot)
    cat = categorize(est, (blo, bhi), (wlo, whi))
    return {'estimate': est, 'boot_lo': blo, 'boot_hi': bhi, 'welch_lo': wlo, 'welch_hi': whi,
            'category': cat, 'category_name': CATEGORY_NAMES[cat],
            'n_seeds': [m.shape[0] for m in sel]}


def standard_contrasts():
    """The pre-registered comparisons S1, S2, S3 and the ablation difference."""
    return {
        'S1 DSRA kazancı (Res2Net): V2-V1': (['PraNet-V2', 'PraNet-V1'], [1, -1]),
        'S1 DSRA kazancı (PVT): V2-V1': (['PVT-PraNet-V2', 'PVT-PraNet-V1'], [1, -1]),
        'S2 etkileşim: (PVT V2-V1)-(Res V2-V1)': (
            ['PVT-PraNet-V2', 'PVT-PraNet-V1', 'PraNet-V2', 'PraNet-V1'], [1, -1, -1, 1]),
        'S3 omurga (RA): PVT-Res': (['PVT-PraNet-V1', 'PraNet-V1'], [1, -1]),
        'S3 omurga (DSRA): PVT-Res': (['PVT-PraNet-V2', 'PraNet-V2'], [1, -1]),
        'S5 tam DSRA - ablasyon (PVT)': (['PVT-PraNet-V2', 'PVT-PraNet-V2-abl'], [1, -1]),
    }


def analyse_dataset(results_root, dataset, protocol='official', n_boot=10000):
    mats, images = load_runs(results_root, dataset, protocol)
    rows = []
    for name, (groups, weights) in standard_contrasts().items():
        if not all(g in mats and mats[g].shape[0] >= 2 for g in groups):
            continue
        r = compare(mats, groups, weights, n_boot)
        r.update({'contrast': name, 'dataset': dataset, 'protocol': protocol})
        rows.append(r)
    return pd.DataFrame(rows), mats


def rho_analysis(mats, n_boot=10000):
    """S5: retained-gain ratio for the PVT backbone, with the guard of H5."""
    need = ['PVT-PraNet-V1', 'PVT-PraNet-V2', 'PVT-PraNet-V2-abl']
    if not all(g in mats and mats[g].shape[0] >= 2 for g in need):
        return None
    gain = compare(mats, ['PVT-PraNet-V2', 'PVT-PraNet-V1'], [1, -1], n_boot)
    diff = compare(mats, ['PVT-PraNet-V2', 'PVT-PraNet-V2-abl'], [1, -1], n_boot)
    out = {'gain_category': gain['category'], 'v2_minus_abl': diff}
    if gain['category'] not in (1, 2):
        out['rho'] = None
        out['H5'] = 'rho hesaplanmadı (PVT kazancı kategori 1/2 değil); mutlak farklarla betimlenir'
        return out
    sel = [mats[g] for g in ['PVT-PraNet-V1', 'PVT-PraNet-V2', 'PVT-PraNet-V2-abl']]
    point = (sel[2].mean() - sel[0].mean()) / (sel[1].mean() - sel[0].mean())
    _, lo, hi = hierarchical_bootstrap(sel, None, n_boot=n_boot,
                                       ratio=([-1, 0, 1], [-1, 1, 0]))
    if hi < 0.5 and diff['welch_lo'] > 0:
        h5 = 'desteklendi'
    elif lo > 0.5:
        h5 = 'reddedildi'
    else:
        h5 = 'belirsiz'
    out.update({'rho': point, 'rho_lo': lo, 'rho_hi': hi, 'H5': h5})
    return out


def variance_decomposition(mats, n_boot=4000):
    """Bootstrap variance of each model's mean when only seeds or only images vary."""
    rows = []
    for c, m in mats.items():
        if m.shape[0] < 2:
            continue
        vs = np.var(hierarchical_bootstrap([m], [1], n_boot=n_boot, mode='seeds')[0])
        vi = np.var(hierarchical_bootstrap([m], [1], n_boot=n_boot, mode='images')[0])
        rows.append({'config': c, 'n_seeds': m.shape[0], 'sd_seeds': np.sqrt(vs),
                     'sd_images': np.sqrt(vi), 'seed_share': vs / (vs + vi)})
    return pd.DataFrame(rows)


def paper_position(mats, paper_values):
    """S4: where the single reported number falls relative to our seed distribution."""
    rows = []
    for config, value in paper_values.items():
        if config not in mats or mats[config].shape[0] < 2:
            continue
        seed_means = mats[config].mean(axis=1)
        lo, hi = prediction_interval(seed_means)
        rows.append({'config': config, 'paper': value, 'mean': seed_means.mean(),
                     'sd': seed_means.std(ddof=1), 'pi_lo': lo, 'pi_hi': hi,
                     'inside': lo <= value <= hi, 'pi_width': hi - lo})
    return pd.DataFrame(rows)


def self_test(seed=0):
    """Sanity checks on synthetic data with a known effect (run before real use)."""
    rng = np.random.default_rng(seed)
    n_img = 100
    img_effect = rng.normal(0, 10, n_img)                     # shared image difficulty
    make = lambda mu, n: mu + img_effect + rng.normal(0, 1, (n, 1)) + rng.normal(0, 2, (n, n_img))
    mats = {'A': make(85, 5), 'B': make(80, 5), 'C': make(85.2, 5)}
    big = compare(mats, ['A', 'B'], [1, -1], n_boot=2000)
    none = compare(mats, ['A', 'C'], [1, -1], n_boot=2000)
    assert big['category'] == 1, big
    assert none['category'] in (3, 4), none
    # pairing check: shared image effect must cancel -> paired CI much narrower than images-only of one model
    _, lo, hi = hierarchical_bootstrap([mats['A'], mats['B']], [1, -1], n_boot=2000, mode='images')
    single = np.std(hierarchical_bootstrap([mats['A']], [1], n_boot=2000, mode='images')[0])
    assert (hi - lo) / 3.92 < 0.5 * single, 'image pairing does not cancel the shared effect'
    # interaction contrast of identical gains should not be category 1
    inter = compare({'a': mats['A'], 'b': mats['B'], 'c': mats['A'] + 0.0, 'd': mats['B'] + 0.0},
                    ['a', 'b', 'c', 'd'], [1, -1, -1, 1], n_boot=2000)
    assert inter['category'] != 1
    lo, hi = prediction_interval([1.0, 2.0, 3.0])
    assert abs((lo + hi) / 2 - 2.0) < 1e-12 and hi - lo > 8      # t(2)=4.30 -> very wide
    return 'self_test OK'


if __name__ == '__main__':
    print(self_test())
