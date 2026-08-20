"""
Selects hard_weight and temperature by VALIDATION AUROC, not test AUROC --
addressing a methodological gap in sweep_hard_weight.py /
compare_curriculum_transitions.py: train.train_model() only ever returns
test AUROC (there's no way to get validation AUROC back out), so those
scripts' "best" value is effectively picked by comparing test-set numbers
across candidates. That's a form of leakage: the test set is meant to be
touched once, to report a final number for an already-decided
configuration, not used to choose between candidates.

This script is a standalone fix that does NOT touch train.py or the
existing sweep scripts at all. It uses its own copy of the sigmoid
curriculum training loop (train_model_with_val) that additionally returns
the best validation AUROC achieved during training -- already computed
internally by train.train_model() for early-stopping model selection, just
never exposed. Selection (which hard_weight / temperature is "best") is
based on validation AUROC; test AUROC is also collected and plotted
alongside for transparency, but never used to choose between candidates.

Unlike sweep_hard_weight.py / compare_curriculum_transitions.py, this uses
a single train/val/test split (same as plot_curvature_vs_loss.py), not
5-fold CV -- the point here is just to demonstrate/confirm that selection
happens on validation rather than test, which a single split shows exactly
as well as 5 folds would, at a fraction of the runtime.

Produces one bar-chart plot per sweep (validation vs. test AUROC per
candidate, with the validation-selected best value marked).
"""
import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

from parms_setting import settings
from data_preprocess import load_data
from instantiation import Create_model
from train import test


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


# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

args = settings()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Load data -- single train/val/test split, same as plot_curvature_vs_loss.py.
# No resampling here, so there's no fold boundary for curvature to fall out
# of sync with: train_loader's attached curvature was already computed from
# exactly these training edges inside load_data().
data_o, train_loader, val_loader, test_loader = load_data(args)

HARD_WEIGHTS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
TEMPERATURES = [0.05, 0.1, 0.2, 0.4, 0.8]
DEFAULT_HARD_WEIGHT = 4.0
DEFAULT_TEMPERATURE = 0.2

hw_val_scores = {}
hw_test_scores = {}
temp_val_scores = {}
temp_test_scores = {}

# hard_weight sweep, temperature fixed at default
for w in HARD_WEIGHTS:
    print(f"[hard_weight sweep] Training hard_weight={w}...")
    model, opt = Create_model(args)
    test_auc, val_auc = train_model_with_val(
        model, opt, data_o, train_loader, val_loader, test_loader, args,
        hard_weight=float(w), temperature=DEFAULT_TEMPERATURE
    )
    hw_val_scores[w] = val_auc
    hw_test_scores[w] = test_auc
    del model, opt

# temperature sweep, hard_weight fixed at default
for g in TEMPERATURES:
    print(f"[temperature sweep] Training temperature={g}...")
    model, opt = Create_model(args)
    test_auc, val_auc = train_model_with_val(
        model, opt, data_o, train_loader, val_loader, test_loader, args,
        hard_weight=DEFAULT_HARD_WEIGHT, temperature=g
    )
    temp_val_scores[g] = val_auc
    temp_test_scores[g] = test_auc
    del model, opt

os.makedirs('finalplots', exist_ok=True)


def report_and_plot(candidates, val_scores, test_scores, xlabel, filename_prefix):
    print("\n" + "=" * 60)
    val_vals = [val_scores[c] for c in candidates]
    test_vals = [test_scores[c] for c in candidates]

    best_idx = int(np.argmax(val_vals))
    best_value = candidates[best_idx]

    for c, v, t in zip(candidates, val_vals, test_vals):
        marker = "  <- selected (best VALIDATION AUROC)" if c == best_value else ""
        print(f"{xlabel}={c}   VAL AUROC: {v:.4f}   TEST AUROC: {t:.4f}{marker}")
    print(f"Selected by validation AUROC: {xlabel}={best_value}")
    print("=" * 60)

    x = np.arange(len(candidates))
    width = 0.35
    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, val_vals, width, label='Validation AUROC', color='steelblue')
    plt.bar(x + width / 2, test_vals, width, label='Test AUROC', color='darkorange')
    plt.xticks(x, [str(c) for c in candidates])
    plt.xlabel(xlabel)
    plt.ylabel('AUROC')
    plt.title(f'Validation-based selection of {xlabel}\n(selected: {xlabel}={best_value}, by validation AUROC, single split)')
    plt.axvline(best_idx, color='green', linestyle='--', alpha=0.5, linewidth=1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f'finalplots/{filename_prefix}_val_vs_test.png'
    plt.savefig(path, dpi=300)
    print(f"Saved plot to {path}")
    return best_value


best_hw = report_and_plot(HARD_WEIGHTS, hw_val_scores, hw_test_scores, 'hard_weight', 'hard_weight_sweep')
best_temp = report_and_plot(TEMPERATURES, temp_val_scores, temp_test_scores, 'temperature', 'temperature_sweep')

print(f"\nFinal selection: hard_weight={best_hw}, temperature={best_temp}")
