"""
5-fold cross-validated hyperparameter selection for hard_weight (lambda)
and temperature (gamma), jointly searched -- built to mirror main.py's
evaluation protocol exactly, so the final reported AUROC is directly
comparable to main.py's number instead of measuring something
structurally different. Curriculum pacing (start_thresh/end_thresh) is
held fixed at train.py's actual hardcoded values, (1.0, 0.0) -- see
train.py's train_model, where start_thresh/end_thresh are plain local
variables, not parameters -- rather than searched. An earlier version of
this script searched pacing jointly too; that's been removed so every
fold's search reflects the exact pacing schedule train.py and main.py
actually run with.

Like main.py, StratifiedKFold rotates over train+val ONLY (not test); the
test set is loaded once and reused UNCHANGED across all 5 folds, exactly
as in main.py. Within each fold, that fold's own train/val split trains
the full joint hyperparameter grid and picks the winner by validation
AUROC (the same val set already used for early-stopping within each
candidate's training run, doing double duty for candidate selection too --
standard practice, and never involves the test set). The winning
configuration is then evaluated once on the shared, fixed test set.
Because test is never touched during the search, this stays leakage-free;
because test is never rotated either, the resulting mean +/- std across
folds is measuring the same quantity as main.py's curriculum_scores, just
with per-fold hyperparameter search replacing main.py's fixed defaults
(hard_weight=6.0, temperature=0.2 -- see train.py's train_model
signature; main.py never overrides hard_weight, so that's the value its
own baseline/curriculum_scores actually ran with).

Hyperparameters searched jointly (not coordinate-wise) per fold:
  - hard_weight in {0, 2, 4, 6, 8}                      (5 candidates)
  - temperature in {0.05, 0.1, 0.2, 0.4, 0.8}           (5 candidates)
5 x 5 = 25 combos/fold x 5 folds = 125 training runs, plus one cheap
test-set evaluation per fold (not a training run) -- roughly 3.5 hours,
scaling from this project's own ~1.7 min/run estimate (see
sweep_hard_weight.py's docstring: 70 runs ~2 hours). Coarsened from a
denser grid after the real (non-synthetic) sweeps in
hard_weight_sweep_val_vs_test.png / temperature_sweep_val_vs_test.png
showed near-flat AUROC across both axes -- a 5-point grid on each still
spans the full range and catches a real effect if one exists, at less
than half the compute of the denser version.

Does not touch train.py or any other existing script.

Produces five figures in finalplots/:
  - nested_cv_selected_hyperparameters.png -- 2 panels: (a) hard_weight
    and (b) temperature selected by each fold's independent search.
  - nested_cv_final_test_auroc.png -- each fold's test AUROC on the SAME
    shared, fixed test set (that fold's own selected-hyperparameter
    model), error bars = bootstrap std of that single estimate; plus the
    mean +/- sample std across the 5 folds -- directly comparable to
    main.py's curriculum_scores mean +/- std, and the headline result.
  - nested_cv_hard_weight_per_fold.png / _temperature_per_fold.png --
    fold x candidate-value "profile" heatmaps: each cell is the best
    validation AUROC achieved at that candidate value, maximized over the
    other hyperparameter (since search is joint, there's no single
    fixed-other-param curve to show -- this is the natural generalization
    of that idea to a joint grid).
  - nested_cv_hard_weight_temperature_joint.png -- the full 2D Hard
    weight x Temperature interaction, mean across the 5 folds.
"""
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Subset, DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from parms_setting import settings
from data_preprocess import load_data, get_curvature, get_proxy_curvature, Data_class
from instantiation import Create_model, set_seed
from train import test


def bootstrap_auroc_std(y_label, y_pred, n_boot=1000, seed=0):
    # Uncertainty of a SINGLE test AUROC estimate (resampling that fold's
    # own test predictions), not fold-to-fold variance -- a
    # legitimate use of the test set since it only describes the
    # already-frozen winning configuration, never selects it.
    y_label = np.asarray(y_label)
    y_pred = np.asarray(y_pred)
    n = len(y_label)
    rng = np.random.default_rng(seed)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yl, yp = y_label[idx], y_pred[idx]
        if len(np.unique(yl)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yl, yp))
    return float(np.std(boot_aucs))

# -- validated palette (see dataviz skill: references/palette.md) --
BLUE = '#2a78d6'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
SURFACE = '#fcfcfb'
NAVY = '#1a3f7a'
FOREST = '#1b5e20'
PLUM = '#4a3aa7'


def build_fold_loader(dataset, idx, edge_to_curv, fold_positive_edges, unique_entity, batch_size, shuffle):
    # Rebuilds (u, v, label) triples plus a *freshly computed* curvature
    # value per sample for this fold, instead of reusing curvature baked in
    # at a single, pre-fold load_data() call. Negative curvature now comes
    # from get_proxy_curvature -- matching main.py's build_fold_loader --
    # instead of a hardcoded inf for every negative pair, so this script's
    # curriculum weighting actually uses negative-edge structure the same
    # way main.py's does (see this file's docstring: the point of this
    # script is to mirror main.py's protocol exactly).

    # nodes with zero edges in THIS fold's training graph -- a negative pair
    # touching one has no real neighborhood for get_proxy_curvature to work
    # from, so it always falls back to the disconnected/inf case. Positive
    # edges can never touch these nodes (zero degree here means it can't be
    # a positive-edge endpoint in this fold), so this only ever drops
    # negatives -- matches main.py's build_fold_loader policy.
    fold_degree = np.zeros(unique_entity, dtype=np.int64)
    if len(fold_positive_edges) > 0:
        np.add.at(fold_degree, fold_positive_edges[:, 0], 1)
        np.add.at(fold_degree, fold_positive_edges[:, 1], 1)
    isolated_this_fold = fold_degree == 0

    triples = []
    curv = []
    neg_pairs = []
    neg_positions = []
    for i in idx:
        label, (u, v), _ = dataset[i]
        u, v, label = int(u), int(v), int(label)
        if label == 0 and (isolated_this_fold[u] or isolated_this_fold[v]):
            # drop rather than train on a fake "confidently easy" value
            continue
        triples.append((u, v, label))
        if label == 1:
            # self-loops get curvature 0 by convention (see get_curvature) --
            # they're deliberately not stored in edge_to_curv
            curv.append(0.0 if u == v else edge_to_curv[tuple(sorted((u, v)))])
        else:
            neg_pairs.append((u, v))
            neg_positions.append(len(curv))
            curv.append(None)

    # proxy curvature for this fold's negatives, computed once for the whole
    # batch against this fold's training graph (same graph edge_to_curv was
    # built from) -- see get_proxy_curvature in data_preprocess.py
    if neg_pairs:
        neg_curv = get_proxy_curvature(unique_entity, fold_positive_edges, np.array(neg_pairs))
        for pos, val in zip(neg_positions, neg_curv.tolist()):
            curv[pos] = val

    triples = np.array(triples, dtype=np.int64)
    curv = torch.tensor(curv, dtype=torch.float)
    return DataLoader(Data_class(triples, curv), batch_size=batch_size, shuffle=shuffle)


def train_and_select(model, optimizer, data_o, train_loader, val_loader, args,
                      use_curvature=False, weight_fct='exp', hard_weight=4.0,
                      temperature=0.2, start_thresh=1.0, end_thresh=0.0):
    # Trains and tracks the best validation-AUROC model state. Never touches
    # test data -- by design, the shared, fixed test set is evaluated
    # exactly once per fold, after the full joint search has already
    # picked a winner (see evaluate_on_test below and the main loop).
    m = torch.nn.Sigmoid()
    loss_no_reduction = torch.nn.BCELoss(reduction='none')
    max_auc = -np.inf
    model_max = copy.deepcopy(model)

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')

    # negative-side pacing, calibrated the same way as train.py's
    # train_model: 5th/95th percentile of this fold's REAL (finite)
    # negative curvature. start_thresh/end_thresh above are fixed constants
    # (positives' proxy curvature has a meaningful [0,1] bound), but
    # negatives' proxy curvature doesn't have a fixed range, so it's
    # calibrated per fold instead of hardcoded -- mirrors train.py exactly
    neg_mask = torch.as_tensor(np.asarray(train_loader.dataset.label)) == 0
    neg_curv = train_loader.dataset.curvature[neg_mask]
    real_neg_curv = neg_curv[torch.isfinite(neg_curv)]
    if len(real_neg_curv) > 0:
        start_thresh_neg = torch.quantile(real_neg_curv, 0.05).item()
        end_thresh_neg = torch.quantile(real_neg_curv, 0.95).item()
    else:
        start_thresh_neg = end_thresh_neg = 0.0

    for epoch in range(args.epochs):
        progress = epoch / (args.epochs - 1) if args.epochs > 1 else 1.0
        current_threshold = start_thresh + progress * (end_thresh - start_thresh)
        current_threshold_neg = start_thresh_neg + progress * (end_thresh_neg - start_thresh_neg)

        for i, (label, inp, curv_batch) in enumerate(train_loader):
            if args.cuda:
                label = label.cuda()
                curv_batch = curv_batch.cuda()
            model.train()
            optimizer.zero_grad()
            output = model(data_o, inp, use_curvature=use_curvature, weight_fct=weight_fct)
            log = torch.squeeze(m(output))
            loss_simple = loss_no_reduction(log, label.float())

            curv_flat = curv_batch.flatten()
            is_pos = label.flatten() == 1
            # positives: hard = low curvature (bottleneck edge)
            pos_hard_mask = torch.sigmoid((current_threshold - curv_flat) / temperature)
            # negatives: hard = high proxy curvature (looks like it should
            # be an edge) -- mirrored, same as train.py's train_model
            neg_hard_mask = torch.sigmoid((curv_flat - current_threshold_neg) / temperature)
            hard_mask = torch.where(is_pos, pos_hard_mask, neg_hard_mask)
            # a non-finite curvature means no real value exists for this
            # pair (disconnected/isolated within this fold) -- treat as no
            # signal, not as inf saturating a mask to 0 or 1
            hard_mask = torch.where(torch.isfinite(curv_flat), hard_mask, torch.zeros_like(hard_mask))
            weights = 1.0 + hard_weight * hard_mask
            loss_train = (loss_simple * weights).mean()

            loss_train.backward()
            optimizer.step()

        if not args.fastmode:
            roc_val, _, _, _, _ = test(model, val_loader, data_o, args, use_curvature=use_curvature, weight_fct=weight_fct, final_train=False)
            if roc_val > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = roc_val
        else:
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    return model_max, max_auc


def evaluate_on_test(model, test_loader, data_o, args, use_curvature=False, weight_fct='exp'):
    # The only place in this script that touches the test set -- called
    # exactly once per fold, on the single winning model.
    auroc_test, _, _, _, details = test(model, test_loader, data_o, args, use_curvature=use_curvature, weight_fct=weight_fct, final_train=True)
    auroc_test_std = bootstrap_auroc_std(details['y_label'], details['y_pred'])
    return auroc_test, auroc_test_std


# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

args = settings()
args.cuda = torch.cuda.is_available()
# fastmode skips per-epoch validation in train_and_select, leaving max_auc
# at -inf for every candidate -- hyperparameter selection below would
# silently degrade to "pick the first candidate" instead of comparing
# validation AUROC. This script's only purpose is that selection, so
# refuse to run rather than fail silently.
assert not args.fastmode, "fastmode disables validation, which breaks hyperparameter selection in this script"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

data_o, train_loader, val_loader, test_loader = load_data(args)

# matches main.py exactly: fold rotation spans train+val only, test_loader
# is loaded once above and reused UNCHANGED across all 5 folds below
combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
all_labels = np.concatenate([
    np.asarray(train_loader.dataset.label),
    np.asarray(val_loader.dataset.label),
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

HARD_WEIGHTS = [0, 2, 4, 6, 8]
TEMPERATURES = [0.05, 0.1, 0.2, 0.4, 0.8]
# Curriculum pacing is held fixed at train.py's actual hardcoded values
# (start_thresh=1.0, end_thresh=0.0 -- see train.py's train_model), not
# searched. train_and_select's own start_thresh/end_thresh defaults below
# already match these exactly, so simply leaving them unset in every call
# below keeps every fold's search on that same pacing schedule.

os.makedirs('finalplots', exist_ok=True)

fold_best_hw = []
fold_best_temp = []
fold_test_auc = []
fold_test_auc_std = []  # bootstrap std of that fold's single test AUROC estimate
fold_search_grids = []  # list of {(hard_weight, temperature): val_auc}, one dict per fold

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(combined_dataset)), all_labels)):

    print(f"\n========== FOLD {fold + 1}/5 ==========")

    # this fold's train/val split IS the search's train/val -- no further
    # inner split needed, since the test set (below) is fixed and shared,
    # never involved in selection at all
    train_edges = []
    for i in train_idx:
        label, (u, v), _ = combined_dataset[i]
        if label == 1:
            train_edges.append((u, v))

    edge_index_fold, edge_cuva_fold, edge_to_curv = get_curvature(
        unique_entity=data_o.num_nodes,
        positive=np.array(train_edges)
    )

    train_loader_fold = build_fold_loader(combined_dataset, train_idx, edge_to_curv, np.array(train_edges), data_o.num_nodes, train_loader.batch_size, shuffle=True)
    # NOTE: unlike train_loader_fold, this reuses combined_dataset's
    # ORIGINAL per-sample curvature (baked in once, at load_data() time),
    # not this fold's edge_to_curv -- deliberately left this way rather
    # than "fixed," because reusing build_fold_loader here would KeyError
    # (edge_to_curv only has entries for train_idx's positive edges;
    # val_idx's positive edges are disjoint from those and were never
    # added to the dict). This is harmless today: CGCN.forward() reads
    # curvature from data_o.curva (current_data.curva below, which IS
    # fold-specific and shared across train/val/test), never from this
    # per-sample field, and test()'s final_train=False path (every
    # validation call) never touches it either.
    val_loader_fold = DataLoader(Subset(combined_dataset, val_idx), batch_size=val_loader.batch_size, shuffle=False)

    current_data = copy.deepcopy(data_o).to(device)
    current_data.edge_index = edge_index_fold.to(device)
    current_data.curva = edge_cuva_fold.to(device)

    # --- full joint grid: hard_weight x temperature, all at once ---
    # test_loader (shared, fixed, loaded once above) is not touched
    # anywhere in this loop -- only once, after the winner is known, below.
    # Pacing is not part of this grid: train_and_select's start_thresh/
    # end_thresh defaults (1.0, 0.0) already match train.py's hardcoded
    # values and are left unset here, so every candidate below trains with
    # that exact pacing schedule.
    total_combos = len(HARD_WEIGHTS) * len(TEMPERATURES)
    print(f"  [fold {fold + 1}] joint grid search ({total_combos} combos)...")
    grid_val = {}
    best_val_auc = -np.inf
    best_combo = None
    best_model = None
    for w in HARD_WEIGHTS:
        for g in TEMPERATURES:
            # explicit reseed so every candidate starts from the
            # identical RNG state -- same initial weights AND the same
            # minibatch shuffle order (train_loader_fold's shuffling
            # draws from the same global generator this resets), so
            # differences in val_auc across candidates reflect the
            # hyperparameters, not random init/ordering noise. This is
            # already implied by Create_model()'s own internal
            # set_seed() call below (nothing between candidates touches
            # the RNG), but made explicit here so it doesn't silently
            # depend on that staying true if this loop is ever edited.
            set_seed(args.seed)
            model, opt = Create_model(args)
            model_max, val_auc = train_and_select(
                model, opt, current_data, train_loader_fold, val_loader_fold, args,
                hard_weight=float(w), temperature=g
            )
            grid_val[(w, g)] = val_auc
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_combo = (w, g)
                if best_model is not None:
                    del best_model
                best_model = model_max
            else:
                del model_max
            del model, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    best_w, best_g = best_combo
    print(f"  [fold {fold + 1}] selected hard_weight={best_w}, temperature={best_g} "
          f"(val AUROC={best_val_auc:.4f})")

    # shared, fixed test_loader touched exactly once here, for the winning
    # model only -- same test set every fold, matching main.py
    test_auc, test_auc_std = evaluate_on_test(best_model, test_loader, current_data, args)
    print(f"  [fold {fold + 1}] test AUROC (shared, fixed test set): "
          f"{test_auc:.4f} (bootstrap std {test_auc_std:.4f})")

    fold_best_hw.append(best_w)
    fold_best_temp.append(best_g)
    fold_test_auc.append(test_auc)
    fold_test_auc_std.append(test_auc_std)
    fold_search_grids.append(grid_val)

    del current_data, best_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------
# Results
# -----------------------
mean_auc = np.mean(fold_test_auc)
std_auc = np.std(fold_test_auc, ddof=1)  # sample std across the 5 fold-specific models

print("\n" + "=" * 60)
for f in range(5):
    print(f"Fold {f + 1}: hard_weight={fold_best_hw[f]}, temperature={fold_best_temp[f]}, "
          f"test AUROC={fold_test_auc[f]:.4f}")
print(f"\nTest AUROC on the shared, fixed test set (mean +/- sample std across the "
      f"5 fold-specific, hyperparameter-searched models) -- directly comparable to "
      f"main.py's curriculum_scores: {mean_auc:.4f} ± {std_auc:.4f}")
print("=" * 60)

os.makedirs('finalplots', exist_ok=True)

# -----------------------
# Figure A: per-fold selected hyperparameters (2 panels)
# -----------------------
plt.rcParams['font.family'] = 'serif'

folds_x = np.arange(1, 6)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color(INK_PRIMARY)
    ax.tick_params(colors=INK_PRIMARY, length=0, labelsize=12)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1.0, linestyle=(0, (1, 2)), zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(folds_x)
    ax.set_xticklabels([str(f) for f in folds_x])
    ax.set_xlabel('Outer fold', color=INK_PRIMARY, fontsize=13)
    ax.set_xlim(0.5, 5.5)

# (a) hard_weight -- linear scale, integer ticks
ax1.plot(folds_x, fold_best_hw, color=NAVY, linewidth=2, marker='o',
          markersize=9, markerfacecolor=NAVY, markeredgecolor=NAVY, zorder=3)
ax1.set_ylabel('Selected Hard weight', color=INK_PRIMARY, fontsize=13)
ax1.set_title('(a) Hard weight', color=INK_PRIMARY, fontsize=15, fontweight='bold', pad=14)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
y_span1 = max(fold_best_hw) - min(fold_best_hw)
label_offset1 = max(y_span1 * 0.08, 0.15)
for x, v in zip(folds_x, fold_best_hw):
    ax1.text(x, v + label_offset1, str(v), ha='center', va='bottom',
              color=NAVY, fontsize=13)
ax1.set_ylim(min(fold_best_hw) - 1, max(fold_best_hw) + 1)

# (b) temperature -- log scale so the doubling steps (0.05..0.8) space evenly
ax2.plot(folds_x, fold_best_temp, color=FOREST, linewidth=2, marker='o',
          markersize=9, markerfacecolor=FOREST, markeredgecolor=FOREST, zorder=3)
ax2.set_ylabel('Selected Temperature', color=INK_PRIMARY, fontsize=13)
ax2.set_title('(b) Temperature', color=INK_PRIMARY, fontsize=15, fontweight='bold', pad=14)
ax2.set_yscale('log')
ax2.set_yticks(TEMPERATURES)
ax2.set_yticklabels([f'{t:.2f}' for t in TEMPERATURES])
ax2.yaxis.set_minor_locator(plt.NullLocator())
for x, v in zip(folds_x, fold_best_temp):
    ax2.text(x, v * 1.3, f'{v:.2f}', ha='center', va='bottom',
              color=FOREST, fontsize=13)
ax2.set_ylim(min(TEMPERATURES) * 0.7, max(TEMPERATURES) * 1.5)

fig.tight_layout()
path_a = 'finalplots/nested_cv_selected_hyperparameters.png'
fig.savefig(path_a, dpi=300, bbox_inches='tight', facecolor=SURFACE)
print(f"\nSaved plot to {path_a}")

plt.rcParams['font.family'] = 'sans-serif'

# -----------------------
# Figure B: test AUROC per fold (same shared, fixed test set every fold,
# each fold's own selected-hyperparameter model) + overall mean. Error
# bars = bootstrap std of each fold's own test AUROC estimate (sampling
# uncertainty of that single number, not fold-to-fold spread, which is
# what the mean +/- std band separately represents). Directly comparable
# to main.py's curriculum_scores mean +/- std -- same test set, same fold
# structure, only difference is per-fold hyperparameter search here vs.
# main.py's fixed defaults.
# -----------------------
fig2, ax = plt.subplots(figsize=(9.5, 4.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['left'].set_color(BASELINE)
ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=11)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=1.0, linestyle=(0, (1, 2)), zorder=0)
ax.set_axisbelow(True)

mean_line = ax.axhline(mean_auc, color=INK_PRIMARY, linestyle='--', linewidth=1.3, zorder=2)
band = ax.axhspan(mean_auc - std_auc, mean_auc + std_auc, color=BLUE, alpha=0.12, zorder=1)

bars = ax.bar(folds_x, fold_test_auc, width=0.55, color=BLUE, zorder=3,
               yerr=fold_test_auc_std, capsize=5,
               error_kw=dict(ecolor=INK_PRIMARY, elinewidth=1.3, capthick=1.3, zorder=4))

bar_lows = [v - e for v, e in zip(fold_test_auc, fold_test_auc_std)]
bar_highs = [v + e for v, e in zip(fold_test_auc, fold_test_auc_std)]
y_low = min(min(bar_lows), mean_auc - std_auc)
y_high = max(max(bar_highs), mean_auc + std_auc)
pad = max((y_high - y_low) * 0.6, 0.01)
ax.set_ylim(y_low - pad, y_high + pad * 1.3)

for rect, v, e in zip(bars, fold_test_auc, fold_test_auc_std):
    ax.text(rect.get_x() + rect.get_width() / 2, v + e + pad * 0.12, f'{v:.4f}',
             ha='center', va='bottom', color=NAVY, fontsize=12, fontweight='bold', zorder=5)

ax.set_xticks(folds_x)
ax.set_xticklabels([f'Fold {f}' for f in folds_x], color=INK_SECONDARY, fontsize=12)
ax.set_xlim(0.4, 5.7)
ax.set_ylabel('Test AUROC', color=INK_SECONDARY, fontsize=12)

ax.legend([mean_line, band], [f'Mean: {mean_auc:.4f}', f'± {std_auc:.4f}'],
          frameon=False, labelcolor=INK_SECONDARY, loc='center left',
          bbox_to_anchor=(1.01, 0.5), fontsize=11)

fig2.tight_layout()
path_b = 'finalplots/nested_cv_final_test_auroc.png'
fig2.savefig(path_b, dpi=300, bbox_inches='tight', facecolor=SURFACE)
print(f"Saved plot to {path_b}")

# -----------------------
# Figures C/D: profile heatmaps, fold x candidate value
# -----------------------
# sequential blue ramp, steps 100->700 (light->dark) -- see dataviz skill
SEQ_BLUE_STEPS = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec',
                   '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
                   '#184f95', '#104281', '#0d366b']
SEQ_BLUE_CMAP = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE_STEPS)


def profile_grid(search_grids, values, combo_index):
    # search is joint (not coordinate-wise), so there's no single "curve
    # at some other fixed value" to plot per fold. Instead, each cell here
    # is the best validation AUROC achieved at that candidate value,
    # maximized over the other hyperparameter -- a profile slice through
    # the joint grid.
    grid = np.zeros((5, len(values)))
    for f in range(5):
        for j, v in enumerate(values):
            matching = [val for combo, val in search_grids[f].items() if combo[combo_index] == v]
            grid[f, j] = max(matching)
    return grid


def plot_value_heatmap(grid, values, fold_selected, xlabel, out_path, value_fmt='{}', row_labels=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(grid, cmap=SEQ_BLUE_CMAP, aspect='auto', origin='lower')
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([value_fmt.format(v) for v in values], color=INK_SECONDARY)
    ax.set_yticks(range(5))
    ax.set_yticklabels(row_labels if row_labels else [f'Fold {f + 1}' for f in range(5)], color=INK_SECONDARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)

    vmin, vmax = grid.min(), grid.max()
    for f in range(5):
        for j, v in enumerate(values):
            val = grid[f, j]
            txt_color = SURFACE if val > (vmin + vmax) / 2 else INK_PRIMARY
            ax.text(j, f, f'{val:.3f}', ha='center', va='center', color=txt_color, fontsize=9)
            if v == fold_selected[f]:
                ax.add_patch(plt.Rectangle((j - 0.5, f - 0.5), 1, 1, fill=False,
                                             edgecolor=INK_PRIMARY, linewidth=2.5, zorder=5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('Best validation AUROC (profile max)', color=INK_SECONDARY)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_MUTED, length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


hw_grid = profile_grid(fold_search_grids, HARD_WEIGHTS, 0)
plot_value_heatmap(
    hw_grid, HARD_WEIGHTS, fold_best_hw,
    'Hard weight (profile max over Temperature)',
    'finalplots/nested_cv_hard_weight_per_fold.png'
)

temp_grid = profile_grid(fold_search_grids, TEMPERATURES, 1)
plot_value_heatmap(
    temp_grid, TEMPERATURES, fold_best_temp,
    'Temperature (profile max over Hard weight)',
    'finalplots/nested_cv_temperature_per_fold.png',
    value_fmt='{:.2f}'
)

# -----------------------
# Figure F: joint Hard weight x Temperature heatmap -- the full 2D
# interaction between these two, which C/D above can't show since each is
# a 1D profile maxed over the other hyperparameter. Cell = mean across the
# 5 folds of validation AUROC at that (hard_weight, temperature) pair --
# no profiling needed since these are the only two searched
# hyperparameters now.
# -----------------------
joint_grid = np.zeros((len(TEMPERATURES), len(HARD_WEIGHTS)))
for ti, g in enumerate(TEMPERATURES):
    for wi, w in enumerate(HARD_WEIGHTS):
        per_fold_vals = [sg[(w, g)] for sg in fold_search_grids]
        joint_grid[ti, wi] = np.mean(per_fold_vals)

fig_f, ax_f = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
ax_f.set_facecolor(SURFACE)
im_f = ax_f.imshow(joint_grid, cmap=SEQ_BLUE_CMAP, aspect='auto', origin='lower')
ax_f.set_xticks(range(len(HARD_WEIGHTS)))
ax_f.set_xticklabels(HARD_WEIGHTS, color=INK_SECONDARY)
ax_f.set_yticks(range(len(TEMPERATURES)))
ax_f.set_yticklabels(TEMPERATURES, color=INK_SECONDARY)
ax_f.set_xlabel('Hard weight', color=INK_SECONDARY)
ax_f.set_ylabel('Temperature', color=INK_SECONDARY)
ax_f.set_title('Validation AUROC (mean across folds)',
                color=INK_PRIMARY, fontsize=12, pad=10)
for spine in ax_f.spines.values():
    spine.set_visible(False)
vmin_f, vmax_f = joint_grid.min(), joint_grid.max()
for ti in range(len(TEMPERATURES)):
    for wi in range(len(HARD_WEIGHTS)):
        val = joint_grid[ti, wi]
        txt_color = SURFACE if val > (vmin_f + vmax_f) / 2 else INK_PRIMARY
        ax_f.text(wi, ti, f'{val:.3f}', ha='center', va='center', color=txt_color, fontsize=9)
cbar_f = fig_f.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
cbar_f.set_label('Validation AUROC', color=INK_SECONDARY)
cbar_f.outline.set_visible(False)
cbar_f.ax.tick_params(colors=INK_MUTED, length=0)

fig_f.tight_layout()
path_f = 'finalplots/nested_cv_hard_weight_temperature_joint.png'
fig_f.savefig(path_f, dpi=300, bbox_inches='tight', facecolor=SURFACE)
print(f"Saved plot to {path_f}")
