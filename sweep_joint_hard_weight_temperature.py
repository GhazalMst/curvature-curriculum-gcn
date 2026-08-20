"""
Joint grid sweep over hard_weight x temperature, evaluated with 5-fold CV,
selection made on VALIDATION AUROC only (test AUROC is also collected and
plotted side-by-side, for transparency, but never used to pick a cell).

This closes the gap the sequential searches in this repo can't: both
sweep_hard_weight.py (hard_weight only, temperature fixed) and
nested_cv_hyperparameter_selection.py's inner search (hard_weight first,
then temperature conditioned on the winner) are coordinate-wise, so neither
can see an interaction between the two knobs -- e.g. a hard_weight that is
only good at a specific temperature would be invisible to both. This script
runs the full joint grid instead: every (hard_weight, temperature) pair,
each trained across all 5 folds.

Unlike nested_cv_hyperparameter_selection.py, this does NOT nest the grid
inside outer folds -- there is no single "selected" hyperparameter or final
leakage-free test number coming out of this script. The point here is
purely diagnostic: visualize how validation (and, side-by-side, test)
AUROC vary across the full 2D space, the way Figure 4c in the Wasserstein-
curriculum paper visualizes accuracy across (temperature, gamma). Reviewer
note: because test AUROC is shown here, it must not be used to justify a
configuration choice on its own -- only the validation panel is
selection-relevant; the test panel is shown for transparency only.

Grid: HARD_WEIGHTS x TEMPERATURES = 9 x 5 = 45 cells x 5 folds = 225 runs.
Does not touch train.py or any other existing script.

Produces one figure: sweep_joint_hard_weight_temperature.png, two heatmap
panels (validation AUROC, test AUROC) sharing one color scale.
"""
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Subset, DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedKFold

from parms_setting import settings
from data_preprocess import load_data, get_curvature, Data_class
from instantiation import Create_model
from train import test

# -- validated palette (see dataviz skill: references/palette.md) --
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
SURFACE = '#fcfcfb'
# sequential blue ramp, steps 100->700 (light->dark), for magnitude encoding
SEQ_BLUE_STEPS = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec',
                   '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
                   '#184f95', '#104281', '#0d366b']
SEQ_BLUE_CMAP = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE_STEPS)


def build_fold_loader(combined_dataset, idx, edge_to_curv, batch_size, shuffle):
    # Rebuilds (u, v, label) triples plus a *freshly computed* curvature
    # value per sample for this fold, instead of reusing curvature baked in
    # at a single, pre-fold load_data() call.
    triples = []
    curv = []
    for i in idx:
        label, (u, v), _ = combined_dataset[i]
        u, v, label = int(u), int(v), int(label)
        triples.append((u, v, label))
        if label == 1:
            # self-loops get curvature 0 by convention (see get_curvature) --
            # they're deliberately not stored in edge_to_curv
            curv.append(0.0 if u == v else edge_to_curv[tuple(sorted((u, v)))])
        else:
            curv.append(float('inf'))
    triples = np.array(triples, dtype=np.int64)
    curv = torch.tensor(curv, dtype=torch.float)
    return DataLoader(Data_class(triples, curv), batch_size=batch_size, shuffle=shuffle)


def train_model_with_val(model, optimizer, data_o, train_loader, val_loader, test_loader, args, use_curvature=False, weight_fct='exp', hard_weight=4.0, temperature=0.2):
    # Sigmoid-curriculum training, matching train.train_model()'s math
    # exactly, but additionally returns the best validation AUROC (max_auc)
    # alongside test AUROC -- train.py itself is untouched.
    m = torch.nn.Sigmoid()
    loss_no_reduction = torch.nn.BCELoss(reduction='none')
    max_auc = 0
    model_max = copy.deepcopy(model)

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')

    start_thresh, end_thresh = 1.0, 0.0
    for epoch in range(args.epochs):
        progress = epoch / (args.epochs - 1) if args.epochs > 1 else 1.0
        current_threshold = start_thresh + progress * (end_thresh - start_thresh)

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
            hard_mask = torch.sigmoid((current_threshold - curv_flat) / temperature)
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

    auroc_test, _, _, _, _ = test(model_max, test_loader, data_o, args, use_curvature=use_curvature, weight_fct=weight_fct, final_train=True)
    return auroc_test, max_auc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    args = settings()
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    data_o, train_loader, val_loader, test_loader = load_data(args)
    combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])

    all_labels = np.concatenate([
        np.asarray(train_loader.dataset.label),
        np.asarray(val_loader.dataset.label)
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_splits = list(skf.split(np.zeros(len(combined_dataset)), all_labels))

    HARD_WEIGHTS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    TEMPERATURES = [0.05, 0.1, 0.2, 0.4, 0.8]

    val_grid = np.zeros((len(TEMPERATURES), len(HARD_WEIGHTS)))
    val_grid_std = np.zeros_like(val_grid)
    test_grid = np.zeros_like(val_grid)
    test_grid_std = np.zeros_like(val_grid)

    # pre-build fold loaders once, reused across every grid cell
    fold_loaders = []
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"Building fold {fold + 1}/5 loaders...")
        train_edges = []
        for i in train_idx:
            label, (u, v), _ = combined_dataset[i]
            if label == 1:
                train_edges.append((u, v))
        edge_index_fold, edge_cuva_fold, edge_to_curv = get_curvature(
            unique_entity=data_o.num_nodes, positive=np.array(train_edges)
        )
        train_loader_fold = build_fold_loader(combined_dataset, train_idx, edge_to_curv, train_loader.batch_size, shuffle=True)
        val_loader_fold = DataLoader(Subset(combined_dataset, val_idx), batch_size=val_loader.batch_size, shuffle=False)
        current_data = copy.deepcopy(data_o).to(device)
        current_data.edge_index = edge_index_fold.to(device)
        current_data.curva = edge_cuva_fold.to(device)
        fold_loaders.append((current_data, train_loader_fold, val_loader_fold))

    total_cells = len(HARD_WEIGHTS) * len(TEMPERATURES)
    for ti, temp in enumerate(TEMPERATURES):
        for hi, hw in enumerate(HARD_WEIGHTS):
            cell_idx = ti * len(HARD_WEIGHTS) + hi + 1
            print(f"\n[{cell_idx}/{total_cells}] hard_weight={hw}, temperature={temp}")
            fold_val, fold_test = [], []
            for fold, (current_data, train_loader_fold, val_loader_fold) in enumerate(fold_loaders):
                model, opt = Create_model(args)
                test_auc, val_auc = train_model_with_val(
                    model, opt, current_data, train_loader_fold, val_loader_fold, test_loader, args,
                    hard_weight=float(hw), temperature=temp
                )
                fold_val.append(val_auc)
                fold_test.append(test_auc)
                del model, opt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            val_grid[ti, hi] = np.mean(fold_val)
            val_grid_std[ti, hi] = np.std(fold_val)
            test_grid[ti, hi] = np.mean(fold_test)
            test_grid_std[ti, hi] = np.std(fold_test)
            print(f"  val AUROC={val_grid[ti, hi]:.4f}+/-{val_grid_std[ti, hi]:.4f}  "
                  f"test AUROC={test_grid[ti, hi]:.4f}+/-{test_grid_std[ti, hi]:.4f}")

    os.makedirs('finalplots', exist_ok=True)
    np.savez('finalplots/joint_sweep_grids.npz',
              hard_weights=HARD_WEIGHTS, temperatures=TEMPERATURES,
              val_grid=val_grid, val_grid_std=val_grid_std,
              test_grid=test_grid, test_grid_std=test_grid_std)

    # shared color scale across both panels so the two files stay visually
    # comparable even though each is saved separately
    vmin = min(val_grid.min(), test_grid.min())
    vmax = max(val_grid.max(), test_grid.max())

    plot_single_heatmap(HARD_WEIGHTS, TEMPERATURES, val_grid, vmin, vmax,
                         'Validation AUROC',
                         'finalplots/sweep_joint_hard_weight_temperature_validation.png')
    plot_single_heatmap(HARD_WEIGHTS, TEMPERATURES, test_grid, vmin, vmax,
                         'Test AUROC',
                         'finalplots/sweep_joint_hard_weight_temperature_test.png')


def plot_single_heatmap(hard_weights, temperatures, grid, vmin, vmax, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(grid, cmap=SEQ_BLUE_CMAP, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    ax.set_xticks(range(len(hard_weights)))
    ax.set_xticklabels(hard_weights, color=INK_SECONDARY)
    ax.set_yticks(range(len(temperatures)))
    ax.set_yticklabels(temperatures, color=INK_SECONDARY)
    ax.set_xlabel('Hard weight', color=INK_SECONDARY)
    ax.set_ylabel('Temperature', color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            txt_color = SURFACE if v > (vmin + vmax) / 2 else INK_PRIMARY
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', color=txt_color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean AUROC', color=INK_SECONDARY)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_MUTED, length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
