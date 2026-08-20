"""
Compares test-set accuracy across curvature ranges for GCN (curvature never
touches the model, use_curvature=False in message passing), with curriculum
learning on vs. off: real (1-hop) Ollivier-Ricci curvature for positive
edges, and proxy curvature (get_proxy_curvature) for negative edges, both
reweighted via the sigmoid-soft threshold schedule -- this calls train.py's
train_model directly (not a reimplementation), so the weighting mechanism
matches main.py exactly. Negative pairs touching a node with zero edges in
train's own positive-edge graph are dropped before training, since there's
no real neighborhood for get_proxy_curvature to compute anything from; any
pair still unreachable within train's graph (a different connected
component) gets curv=inf, which train_model's isfinite check treats as no
signal rather than a fake "confidently easy" value.

Both models are trained on the same single train/val/test split (not
cross-validated -- this is for visual inspection, not a significance test).
For each, we collect (curvature, correct/incorrect) for every real (positive)
test edge and bin by curvature range. The headline output is a diff heatmap:
accuracy(+CL) - accuracy(no CL) in every curvature bin, so it's immediately
visible where curriculum learning helps vs. hurts.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torch.utils.data import DataLoader

from parms_setting import settings
from data_preprocess import load_data, Data_class
from instantiation import Create_model
from train import train_model

args = settings()
args.cuda = torch.cuda.is_available()
dataset_name = Path(args.in_file).stem
# NOTE: args.correlation is intentionally left at its default (False) --
# test()'s correlation block references an undefined `df` and will crash
# with a NameError if args.correlation is True.

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

print(f"Device: {'cuda' if args.cuda else 'cpu'}")

data_o, train_loader, val_loader, test_loader = load_data(args)

# nodes with zero edges in train's own positive-edge graph -- a negative
# pair touching one has no real neighborhood for get_proxy_curvature to
# work from, so it always falls back to the disconnected/inf case. Positive
# edges can never touch these nodes (zero degree here means it can't be a
# positive-edge endpoint in train), so this only ever drops negatives --
# matches main.py's build_fold_loader policy.
train_label = np.asarray(train_loader.dataset.label)
train_e1 = np.asarray(train_loader.dataset.entity1)
train_e2 = np.asarray(train_loader.dataset.entity2)
train_pos_mask = train_label == 1
train_degree = np.zeros(data_o.num_nodes, dtype=np.int64)
np.add.at(train_degree, train_e1[train_pos_mask], 1)
np.add.at(train_degree, train_e2[train_pos_mask], 1)
isolated_in_train = train_degree == 0
keep_mask = train_pos_mask | ~(isolated_in_train[train_e1] | isolated_in_train[train_e2])

if not keep_mask.all():
    print(f"Dropping {int((~keep_mask).sum())} negative training pair(s) touching a node "
          f"isolated in train's positive-edge graph")
    filtered_triples = np.stack(
        [train_e1[keep_mask], train_e2[keep_mask], train_label[keep_mask]], axis=1
    ).astype(np.int64)
    filtered_curv = train_loader.dataset.curvature[torch.as_tensor(keep_mask)]
    train_loader = DataLoader(
        Data_class(filtered_triples, filtered_curv),
        batch_size=train_loader.batch_size, shuffle=True
    )


def run_and_collect(curriculum, use_curvature):
    model, opt = Create_model(args)
    auroc, details = train_model(
        model, opt, data_o, train_loader, val_loader, test_loader, args,
        curriculum=curriculum, use_curvature=use_curvature,
        return_details=True
    )
    y_curv, y_label, y_pred = details['y_curv'], details['y_label'], details['y_pred']

    df = pd.DataFrame({
        'curvature': np.asarray(y_curv, dtype=np.float64),
        'label': np.asarray(y_label),
        'pred': np.asarray(y_pred, dtype=np.float64),
    })
    df['correct'] = (df['pred'].round() == df['label']).astype(float)
    df = df[df['label'] == 1]              # real edges only -- negatives now carry generalized
                                            # curvature too, but this analysis is about real edges
    df = df[np.isfinite(df['curvature'])]  # drop the rare disconnected-pair inf fallback, if any
    return auroc, df


print("Training GCN (no curriculum)...")
auroc_gcn_base, df_gcn_base = run_and_collect(curriculum=False, use_curvature=False)
print(f"GCN (no curriculum) test AUROC: {auroc_gcn_base:.4f}")

print("\nTraining GCN (curriculum)...")
auroc_gcn_curr, df_gcn_curr = run_and_collect(curriculum=True, use_curvature=False)
print(f"GCN (curriculum) test AUROC: {auroc_gcn_curr:.4f}")


# -----------------------
# Bin by curvature range -- shared bins across both models (the test edges
# and their curvature values are identical regardless of which model
# evaluates them, so bin counts are identical across both too). Join on the
# integer bin index, not the float midpoint -- pd.cut's Interval.mid gets
# internally rounded to fewer decimal places than our own bin_mids
# computation, so reindexing by float value silently matches nothing.
# -----------------------
N_BINS = 40  # finer curvature resolution -- check the bin-count plot before
             # trusting any single point, since narrower bins mean fewer
             # edges (and noisier estimates) per bin, especially in the tails
all_dfs = [df_gcn_base, df_gcn_curr]
lo = min(df['curvature'].min() for df in all_dfs)
hi = max(df['curvature'].max() for df in all_dfs)
bin_edges = np.linspace(lo, hi, N_BINS + 1)
bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_labels = [f"[{bin_edges[i]: .3f}, {bin_edges[i + 1]: .3f})" for i in range(N_BINS)]

for df in all_dfs:
    df['bin_idx'] = pd.cut(df['curvature'], bins=bin_edges, labels=False)


def bin_stats(df, metric):
    binned = df.groupby('bin_idx', observed=True)[metric].agg(['mean', 'count'])
    binned = binned.reindex(range(N_BINS))
    binned.index = bin_mids
    return binned


os.makedirs('finalplots', exist_ok=True)


def print_bin_summary(binned, metric_name):
    for label, (mid, row) in zip(bin_labels, binned.iterrows()):
        if pd.notna(row['mean']):
            print(f"  {label}: n={int(row['count']):5d}  {metric_name}={row['mean']:.4f}")
        else:
            print(f"  {label}: n=0")


def make_line_plot(base_indexed, curr_indexed, base_label, curr_label, base_auroc, curr_auroc, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(bin_mids, base_indexed['mean'], marker='o', label=f'{base_label}, AUROC={base_auroc:.3f}')
    plt.plot(bin_mids, curr_indexed['mean'], marker='o', label=f'{curr_label}, AUROC={curr_auroc:.3f}')
    plt.xlabel('Curvature')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = f'finalplots/{filename}'
    plt.savefig(path, dpi=300)
    print(f"Saved plot to {path}")


def make_diff_bar_chart(base_indexed, curr_indexed, title, filename, cmap='RdYlGn'):
    # diff > 0 (green): curriculum learning improved accuracy in this bin.
    # diff < 0 (red): curriculum learning hurt accuracy in this bin.
    # Empty bins (n=0) are simply skipped -- there's nothing to plot there.
    diff = (curr_indexed['mean'] - base_indexed['mean']).values
    counts = base_indexed['count'].values
    valid = ~np.isnan(diff)

    norm = plt.Normalize(vmin=-np.nanmax(np.abs(diff)), vmax=np.nanmax(np.abs(diff)))
    colors = plt.get_cmap(cmap)(norm(diff[valid]))

    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        bin_mids[valid], diff[valid],
        width=(hi - lo) / N_BINS * 0.9,
        color=colors, edgecolor='black', linewidth=0.3
    )
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel('Curvature interval')
    plt.ylabel('Δ accuracy (+CL − no CL)')
    plt.title(title)
    plt.grid(alpha=0.3, axis='y')

    # label each bar with its sample count, so a tall bar backed by n=2
    # isn't mistaken for a reliable result
    for x, y, n in zip(bin_mids[valid], diff[valid], counts[valid]):
        offset = 0.01 if y >= 0 else -0.01
        va = 'bottom' if y >= 0 else 'top'
        plt.text(x, y + offset, f'n={int(n)}', ha='center', va=va, fontsize=6, rotation=90)

    plt.tight_layout()
    path = f'finalplots/{filename}'
    plt.savefig(path, dpi=300)
    print(f"Saved diff bar chart to {path}")


def make_combined_heatmap(base_indexed, curr_indexed, base_label, curr_label, title, subtitle, filename,
                           acc_cmap='YlGnBu', diff_cmap='RdYlGn'):
    # Two panels sharing the same curvature-interval rows: left = accuracy
    # for both conditions (0-1 scale), right = the diff between them
    # (diverging scale, centered at 0). Kept as two color scales, not one --
    # the diff's range (~-0.5 to +0.15) would be unreadable on a 0-1 scale.
    diff = (curr_indexed['mean'] - base_indexed['mean']).values
    counts = base_indexed['count'].values

    # bin_labels is ordered low-to-high (most negative curvature first), and
    # seaborn/pandas render row 0 at the top -- so building straight from
    # bin_labels (no reversal) already puts most-negative curvature at top,
    # matching the label below. A previous version of this code reversed the
    # row order here while keeping this same comment, which silently put the
    # *most positive* curvature at the top instead -- the opposite of what
    # was intended.
    acc_df = pd.DataFrame({
        base_label: base_indexed['mean'].values,
        curr_label: curr_indexed['mean'].values,
    }, index=bin_labels)  # most negative curvature at top

    diff_df = pd.DataFrame({
        'Accuracy Improvement\n(+CL − no CL)': diff,
    }, index=bin_labels)

    counts_rev = counts

    acc_annot = acc_df.astype(object).copy()
    for col in acc_df.columns:
        for idx, n_val in zip(acc_df.index, counts_rev):
            val = acc_df.loc[idx, col]
            acc_annot.loc[idx, col] = "— (n=0)" if pd.isna(val) else f"{val:.2f} (n={int(n_val)})"

    diff_annot = diff_df.astype(object).copy()
    for col in diff_df.columns:
        for idx, n_val in zip(diff_df.index, counts_rev):
            val = diff_df.loc[idx, col]
            diff_annot.loc[idx, col] = "— (n=0)" if pd.isna(val) else f"{val:+.3f} (n={int(n_val)})"

    # NOTE: no sharey=True here -- matplotlib auto-hides "redundant" tick
    # labels across shared axes, which was silently wiping out ax1's
    # curvature-interval labels too (not just ax2's), leaving the y-axis
    # blank. ax2.set_ylim(...) below keeps the two panels row-aligned instead.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 13),
        gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.1}
    )

    sns.heatmap(
        acc_df.astype(float), annot=acc_annot, fmt='', cmap=acc_cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor='white', cbar_kws={'label': 'Accuracy'}, ax=ax1,
        annot_kws={'fontsize': 7}, yticklabels=1
    )
    ax1.xaxis.tick_top()
    ax1.set_xlabel('')
    ax1.set_ylabel('Curvature Interval')
    ax1.tick_params(axis='y', labelsize=7, rotation=0)

    sns.heatmap(
        diff_df.astype(float), annot=diff_annot, fmt='', cmap=diff_cmap, center=0,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Accuracy Improvement (+CL − no CL)'}, ax=ax2,
        annot_kws={'fontsize': 7}, yticklabels=1
    )
    ax2.xaxis.tick_top()
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.tick_params(left=False)
    ax2.set_yticklabels([])
    ax2.set_ylim(ax1.get_ylim())

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    # anchor to the axes' actual bottom edge, not a guessed figure-fraction
    # value -- matplotlib reserves a default margin below the axes, so a
    # fixed y like -0.005 ends up far below the heatmap, not just below it
    axes_bottom = min(ax1.get_position().y0, ax2.get_position().y0)
    fig.text(0.5, axes_bottom - 0.015, subtitle, fontsize=11, ha='center', color='dimgray')
    path = f'finalplots/{filename}'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved combined heatmap to {path}")


def analyze_family(family_name, prefix, base_label, base_auroc, df_base, curr_label, curr_auroc, df_curr):
    base_acc = bin_stats(df_base, 'correct')
    curr_acc = bin_stats(df_curr, 'correct')

    print(f"\n=== [{family_name}] Per-bin summary: accuracy (n edges) ===")
    print(f"{base_label}:")
    print_bin_summary(base_acc, 'accuracy')
    print(f"{curr_label}:")
    print_bin_summary(curr_acc, 'accuracy')

    make_line_plot(base_acc, curr_acc, base_label, curr_label, base_auroc, curr_auroc,
                    'Test Accuracy',
                    'Effect of Curriculum Learning on GCN performance Across Curvature',
                    f'{prefix}_curvature_vs_accuracy_comparison.png')

    make_diff_bar_chart(base_acc, curr_acc,
                       f'Accuracy change from curriculum learning:\n{family_name}',
                       f'{prefix}_curvature_vs_accuracy_heatmap.png')

    family_short = prefix.upper()
    make_combined_heatmap(
        base_acc, curr_acc,
        f'Accuracy Without\nCurriculum Learning ({family_short})',
        f'Accuracy With\nCurriculum Learning ({family_short})',
        'Model Accuracy With and Without Curriculum Learning',
        f'Dataset name: {dataset_name}',
        f'{prefix}_curvature_combined_heatmap.png'
    )

    # The line plot can't label every point with its interval without
    # becoming unreadable (40 bins) -- this CSV has the exact interval and
    # diff for every point, in the same left-to-right order.
    details_df = pd.DataFrame({
        'curvature_interval': bin_labels,
        'curvature_bin_mid': bin_mids,
        'n_edges': base_acc['count'].values,
        f'{base_label}_accuracy': base_acc['mean'].values,
        f'{curr_label}_accuracy': curr_acc['mean'].values,
        'accuracy_diff_plusCL_minus_noCL': (curr_acc['mean'] - base_acc['mean']).values,
    })
    csv_path = f'finalplots/{prefix}_bin_details.csv'
    details_df.to_csv(csv_path, index=False)
    print(f"Saved bin details (interval -> value mapping) to {csv_path}")

    return base_acc  # for the shared bin-count plot (counts are identical across both models)


gcn_counts = analyze_family(
    'GCN: with vs. without curriculum learning', 'gcn',
    'GCN(without CL)', auroc_gcn_base, df_gcn_base,
    'GCN(with CL)', auroc_gcn_curr, df_gcn_curr
)

# -----------------------
# Sample counts (shared bins -- identical across both models, since it's
# the same fixed test edges every time, just evaluated by different models)
# -----------------------
plt.figure(figsize=(10, 4))
plt.bar(bin_mids, gcn_counts['count'].fillna(0), width=(hi - lo) / N_BINS * 0.9, alpha=0.6, label='Test edges per bin')
plt.xlabel('Curvature')
plt.ylabel('Number of real test edges in bin')
plt.title('Sample count per curvature bin (same bins/edges used for every plot above)')
plt.legend()
plt.tight_layout()
counts_path = 'finalplots/curvature_bin_counts.png'
plt.savefig(counts_path, dpi=300)
print(f"\nSaved bin-count plot to {counts_path}")
