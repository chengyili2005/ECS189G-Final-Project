import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
LATEST_HOTPOT = 'SmolLM2-1.7B-InfoRAG/SmolLM2-1.7B_hotpot_final.csv'
LATEST_WOW    = 'SmolLM2-1.7B-InfoRAG/SmolLM2-1.7B_wow_final.csv'

PALETTE = {
    'Qwen2.5':   '#4E79A7',
    'Llama-3.2': '#F28E2B',
    'gemma-2':   '#E15759',
    'gemma':     '#76B7B2',
    'phi-2':     '#59A14F',
    'SmolLM2':   '#B07AA1',
}

STEPS = [1000, 2000, 3000, 4000, 5000]

# ── Load & Parse ─────────────────────────────────────────────────────────────
def load(path):
    return pd.read_csv(path, index_col=0).T

hotpot_df = load(LATEST_HOTPOT)
wow_df    = load(LATEST_WOW)

def parse_index(df, benchmark):
    records = []
    for idx in df.index:
        name = idx.replace(f'-{benchmark}', '')
        parts = name.split('-')
        if parts[0] == 'base':
            model_str = '-'.join(parts[1:])
            steps = 0
        else:  # tuned
            steps = int(parts[-1])
            model_str = '-'.join(parts[1:-1])

        # Identify family and param size
        if model_str.startswith('Qwen2.5'):
            family = 'Qwen2.5'
            size = model_str.split('-')[-1] if model_str != 'Qwen2.5' else None
        elif model_str.startswith('Llama-3.2'):
            family = 'Llama-3.2'
            size = model_str.split('-')[-1]
        elif model_str.startswith('gemma-2-'):
            family = 'gemma-2'
            size = model_str.split('-')[-1]
        elif model_str.startswith('gemma-'):
            family = 'gemma'
            size = model_str.split('-')[-1]
        elif model_str.startswith('phi-2'):
            family = 'phi-2'
            size = '2.7B'
        elif model_str.startswith('SmolLM2'):
            family = 'SmolLM2'
            size = '1.7B'
        else:
            family = model_str
            size = None

        records.append({
            'idx': idx,
            'family': family,
            'size': size,
            'model_str': model_str,
            'steps': steps,
            'is_base': steps == 0,
            'em_score': float(df.loc[idx, 'em_score']),
            'f1_score': float(df.loc[idx, 'f1_score']),
        })
    return pd.DataFrame(records)

hp = parse_index(hotpot_df, 'hotpot')
ww = parse_index(wow_df, 'wow')

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_color(family):
    return PALETTE.get(family, '#999999')

def param_to_float(s):
    """Convert param string like '3B' -> 3.0"""
    if s is None:
        return np.nan
    s = s.upper().replace('B','').replace('M','e-3')
    try:
        return float(s)
    except:
        return np.nan

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Step size vs performance (F1 & EM), per family
# ═══════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
fig1.suptitle('Effect of Fine-tuning Steps on Benchmark Performance',
              fontsize=15, fontweight='bold', y=1.01)

datasets = [('HotpotQA', hp), ('WoW', ww)]
metrics  = [('f1_score', 'F1 Score'), ('em_score', 'Exact Match')]

for col, (bench_name, df) in enumerate(datasets):
    for row, (metric, metric_label) in enumerate(metrics):
        ax = axes[row][col]
        ax.set_title(f'{bench_name} – {metric_label}', fontweight='bold')

        for family, grp in df.groupby('family'):
            # Average across param sizes at each step count
            tuned = grp[~grp['is_base']].groupby('steps')[metric].mean()
            base_val = grp[grp['is_base']][metric].mean()

            xs = [0] + list(tuned.index)
            ys = [base_val] + list(tuned.values)

            color = get_color(family)
            ax.plot(xs, ys, marker='o', color=color, linewidth=2,
                    label=family, markersize=5)
            ax.axhline(base_val, color=color, linestyle=':', alpha=0.4, linewidth=1)

        ax.set_xlabel('Fine-tuning Steps')
        ax.set_ylabel(metric_label)
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.set_xticklabels(['base', '1k', '2k', '3k', '4k', '5k'])
        ax.grid(True, alpha=0.3)
        if row == 0 and col == 1:
            ax.legend(fontsize=8, loc='upper right')

plt.savefig('fig1_steps_vs_performance.png', dpi=150, bbox_inches='tight')
print("Saved fig1_steps_vs_performance.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Parameter size effect (families with multiple sizes)
# ═══════════════════════════════════════════════════════════════════════════
multi_size_families = ['Qwen2.5', 'Llama-3.2']  # have multiple param counts

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
fig2.suptitle('Effect of Parameter Size on Benchmark Performance\n(families with multiple sizes)',
              fontsize=14, fontweight='bold')

SIZE_MARKERS = {'0.5B': 's', '1B': 'D', '1.5B': '^', '3B': 'o', '2B': 'P'}

for col, (bench_name, df) in enumerate(datasets):
    for row, (metric, metric_label) in enumerate(metrics):
        ax = axes2[row][col]
        ax.set_title(f'{bench_name} – {metric_label}', fontweight='bold')

        for family in multi_size_families:
            fgrp = df[df['family'] == family]
            color = get_color(family)
            sizes_present = sorted(fgrp['size'].dropna().unique(),
                                   key=lambda x: param_to_float(x))
            for sz in sizes_present:
                sgrp = fgrp[fgrp['size'] == sz]
                tuned = sgrp[~sgrp['is_base']].groupby('steps')[metric].mean()
                base_val = sgrp[sgrp['is_base']][metric].mean()
                xs = [0] + list(tuned.index)
                ys = [base_val] + list(tuned.values)
                marker = SIZE_MARKERS.get(sz, 'o')
                shade = 0.4 + 0.6 * (param_to_float(sz) / 3.5)
                r, g, b, _ = plt.matplotlib.colors.to_rgba(color)
                c = (r * shade, g * shade, b * shade)
                ax.plot(xs, ys, marker=marker, color=c, linewidth=2,
                        label=f'{family}-{sz}', markersize=6)

        ax.set_xlabel('Fine-tuning Steps')
        ax.set_ylabel(metric_label)
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.set_xticklabels(['base', '1k', '2k', '3k', '4k', '5k'])
        ax.grid(True, alpha=0.3)
        if row == 0 and col == 1:
            ax.legend(fontsize=8, loc='best')

plt.savefig('fig2_param_size_effect.png', dpi=150, bbox_inches='tight')
print("Saved fig2_param_size_effect.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Heatmap of best improvement (base → best tuned) per model
# ═══════════════════════════════════════════════════════════════════════════
def compute_improvement(df):
    rows = []
    for model_str, grp in df.groupby('model_str'):
        base = grp[grp['is_base']]
        tuned = grp[~grp['is_base']]
        if base.empty or tuned.empty:
            continue
        best_f1 = tuned['f1_score'].max()
        base_f1 = base['f1_score'].values[0]
        best_em = tuned['em_score'].max()
        base_em = base['em_score'].values[0]
        rows.append({
            'model': model_str,
            'delta_f1': best_f1 - base_f1,
            'delta_em': best_em - base_em,
        })
    return pd.DataFrame(rows).set_index('model')

imp_hp = compute_improvement(hp)
imp_ww = compute_improvement(ww)

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
fig3.suptitle('Maximum Improvement from Fine-tuning (Base → Best Tuned)',
              fontsize=14, fontweight='bold')

for ax, (bench_name, imp) in zip(axes3, [('HotpotQA', imp_hp), ('WoW', imp_ww)]):
    data = imp[['delta_f1', 'delta_em']].sort_values('delta_f1', ascending=False)
    im = ax.imshow(data.values.T, aspect='auto',
                   cmap='RdYlGn', vmin=-0.1, vmax=0.3)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['ΔF1', 'ΔEM'])
    ax.set_title(bench_name, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score improvement')
    for i, row in enumerate(data.values):
        for j, val in enumerate(row):
            ax.text(i, j, f'{val:+.3f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(val) < 0.15 else 'white')

plt.savefig('fig3_improvement_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved fig3_improvement_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Statistical Significance: paired t-test base vs all tuned
# ═══════════════════════════════════════════════════════════════════════════
def stat_test(df, bench_name):
    """
    For each family: pair base score vs scores at each step size,
    run one-sample t-test (are improvements > 0?) and paired t-test.
    Returns a summary dataframe.
    """
    results = []
    for family, grp in df.groupby('family'):
        base_scores_f1 = grp[grp['is_base']]['f1_score'].values
        tuned_scores_f1 = grp[~grp['is_base']]['f1_score'].values
        base_scores_em = grp[grp['is_base']]['em_score'].values
        tuned_scores_em = grp[~grp['is_base']]['em_score'].values

        # Difference distribution: each tuned - mean(base)
        deltas_f1 = tuned_scores_f1 - base_scores_f1.mean()
        deltas_em = tuned_scores_em - base_scores_em.mean()

        t_f1, p_f1 = stats.ttest_1samp(deltas_f1, 0)
        t_em, p_em = stats.ttest_1samp(deltas_em, 0)

        results.append({
            'family': family,
            'bench': bench_name,
            'mean_delta_f1': deltas_f1.mean(),
            'p_f1': p_f1,
            'sig_f1': '***' if p_f1 < 0.001 else ('**' if p_f1 < 0.01 else ('*' if p_f1 < 0.05 else 'ns')),
            'mean_delta_em': deltas_em.mean(),
            'p_em': p_em,
            'sig_em': '***' if p_em < 0.001 else ('**' if p_em < 0.01 else ('*' if p_em < 0.05 else 'ns')),
        })
    return pd.DataFrame(results)

stat_hp = stat_test(hp, 'HotpotQA')
stat_ww = stat_test(ww, 'WoW')
stat_all = pd.concat([stat_hp, stat_ww])

print("\n── Statistical Significance (one-sample t-test: mean improvement > 0) ──")
print(stat_all.to_string(index=False))

# Plot significance summary
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
fig4.suptitle('Statistical Significance of Fine-tuning Improvement\n(one-sample t-test vs. baseline; * p<.05, ** p<.01, *** p<.001)',
              fontsize=13, fontweight='bold')

for ax, (bench_name, stat_df) in zip(axes4, [('HotpotQA', stat_hp), ('WoW', stat_ww)]):
    families = stat_df['family'].values
    x = np.arange(len(families))
    w = 0.35

    bars_f1 = ax.bar(x - w/2, stat_df['mean_delta_f1'], w,
                     label='ΔF1', color=[get_color(f) for f in families], alpha=0.9)
    bars_em = ax.bar(x + w/2, stat_df['mean_delta_em'], w,
                     label='ΔEM', color=[get_color(f) for f in families], alpha=0.5,
                     edgecolor=[get_color(f) for f in families], linewidth=1.5)

    # Annotate significance
    for i, (_, row) in enumerate(stat_df.iterrows()):
        ax.text(i - w/2, row['mean_delta_f1'] + 0.002, row['sig_f1'],
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(i + w/2, row['mean_delta_em'] + 0.002, row['sig_em'],
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(bench_name, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha='right')
    ax.set_ylabel('Mean Score Improvement (tuned − base)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.savefig('fig4_statistical_significance.png', dpi=150, bbox_inches='tight')
print("Saved fig4_statistical_significance.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Step-by-step progression per family (small multiples)
# ═══════════════════════════════════════════════════════════════════════════
families = sorted(hp['family'].unique())
n = len(families)
fig5, axes5 = plt.subplots(2, n, figsize=(4*n, 8), constrained_layout=True)
fig5.suptitle('Step-by-Step F1 Progression per Model Family\n(solid = HotpotQA, dashed = WoW)',
              fontsize=13, fontweight='bold')

SIZE_ORDER = ['0.5B', '1B', '1.5B', '1.7B', '2B', '2.7B', '3B']

for col, family in enumerate(families):
    for row, metric in enumerate(['f1_score', 'em_score']):
        ax = axes5[row][col]
        ax.set_title(family if row == 0 else '', fontweight='bold')

        for df, ls, bench in [(hp, '-', 'HotpotQA'), (ww, '--', 'WoW')]:
            fgrp = df[df['family'] == family]
            sizes = fgrp['size'].dropna().unique()
            if len(sizes) == 0:
                sizes = [None]
            sizes = sorted(sizes, key=lambda x: SIZE_ORDER.index(x) if x in SIZE_ORDER else 99)

            for sz in sizes:
                if sz is None:
                    sgrp = fgrp
                else:
                    sgrp = fgrp[fgrp['size'] == sz]
                tuned = sgrp[~sgrp['is_base']].groupby('steps')[metric].mean()
                base_val = sgrp[sgrp['is_base']][metric].mean()
                xs = [0] + list(tuned.index)
                ys = [base_val] + list(tuned.values)
                lbl = f'{sz}' if sz else bench
                ax.plot(xs, ys, linestyle=ls, marker='o', markersize=4,
                        color=get_color(family),
                        alpha=0.5 + 0.5 * (SIZE_ORDER.index(sz)/len(SIZE_ORDER) if sz in SIZE_ORDER else 0.5),
                        label=lbl)

        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.set_xticklabels(['B','1k','2k','3k','4k','5k'], fontsize=7)
        ax.set_ylabel('F1' if metric == 'f1_score' else 'EM', fontsize=8)
        ax.grid(True, alpha=0.3)
        if col == 0 and row == 0:
            ax.legend(fontsize=7)

plt.savefig('fig5_family_progression.png', dpi=150, bbox_inches='tight')
print("Saved fig5_family_progression.png")

plt.show()
print("\nAll figures saved. ✓")