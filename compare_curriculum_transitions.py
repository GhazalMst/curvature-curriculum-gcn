"""
Compares four curriculum-learning weighting schemes on whatever dataset
--in_file points to, all using the same threshold schedule and hard_weight,
differing only in how the hard/easy transition around the threshold is
shaped:
  - hard:     the original hard cutoff -- an edge gets the full hard_weight
              boost the instant its curvature crosses the (epoch-dependent)
              threshold, nothing otherwise.
  - sigmoid:  the current default in train.train_model() -- the same
              threshold, but the hard/easy boundary is smoothed with a
              sigmoid of width `temperature` instead of jumping instantly.
  - erf:      same idea as sigmoid but with a Gaussian-error-function
              transition, which has faster-decaying tails than sigmoid.
  - ensemble: takes the elementwise maximum of the sigmoid and erf masks at
              each threshold -- whichever of the two currently signals
              "harder" wins, rather than averaging them down.

hard, erf, and ensemble are reimplemented here as train_model_variant(),
since train.py only supports sigmoid -- train.py itself is intentionally
left untouched. sigmoid itself is trained via train.train_model() directly.

hard_weight=4.0 and temperature=0.2 are held fixed at the values already
validated as best, so this isolates the effect of the transition shape
alone. Uses the same 5-fold CV setup as main.py (stratified by label,
training curvature recomputed fresh per fold, no validation curvature
computed since it's never consumed), and mirrors main.py's output style:
per-fold progress prints (no plots), then a final mean +/- std AUC summary
for each config.
"""
import copy
import math
import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset, DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedKFold

from parms_setting import settings
from data_preprocess import load_data, get_curvature, Data_class
from instantiation import Create_model
from train import train_model, test


def build_fold_loader(combined_dataset, idx, edge_to_curv, batch_size, shuffle):
    # Rebuilds (u, v, label) triples plus a *freshly computed* curvature
    # value per sample for this fold, instead of reusing the curvature
    # baked into combined_dataset at the single, pre-fold load_data() call --
    # that stale curvature doesn't respect the current fold's train/val
    # boundary, since it was computed once from a different, original split.
    triples = []
    curv = []
    for i in idx:
        label, (u, v), _ = combined_dataset[i]
        u, v, label = int(u), int(v), int(label)
        triples.append((u, v, label))
        if label == 1:
            # self-loops (a node interacting with itself) get curvature 0 by
            # convention, same as get_curvature() -- they're deliberately not
            # stored in edge_to_curv, so looking one up there would KeyError
            curv.append(0.0 if u == v else edge_to_curv[tuple(sorted((u, v)))])
        else:
            curv.append(float('inf'))
    triples = np.array(triples, dtype=np.int64)
    curv = torch.tensor(curv, dtype=torch.float)
    return DataLoader(Data_class(triples, curv), batch_size=batch_size, shuffle=shuffle)


def _mask_for(mode, current_threshold, curv_flat, temperature):
    if mode == 'hard':
        # hard cutoff: full weight boost the instant curv crosses the threshold
        return (curv_flat <= current_threshold).float()
    if mode == 'erf':
        # Gaussian-CDF S-curve: same shape family as sigmoid but with
        # faster-decaying tails, so it saturates to 0/1 sooner on either
        # side of the threshold instead of sigmoid's long exponential tail
        return 0.5 * (1.0 + torch.erf((current_threshold - curv_flat) / (temperature * math.sqrt(2))))
    if mode == 'ensemble':
        # take whichever of sigmoid/erf currently signals "harder" at this
        # curvature, rather than averaging them down
        sigmoid_mask = torch.sigmoid((current_threshold - curv_flat) / temperature)
        erf_mask = _mask_for('erf', current_threshold, curv_flat, temperature)
        return torch.max(sigmoid_mask, erf_mask)
    raise ValueError(f"Unknown mode: {mode!r}")


def train_model_variant(mode, model, optimizer, data_o, train_loader, val_loader, test_loader, args, use_curvature=True, weight_fct='exp', curriculum=False, hard_weight=4.0, temperature=0.2, documentation=False, return_details=False):
    # Identical to train.train_model() except the mask is selected by `mode`
    # (hard / erf / ensemble) instead of always being sigmoid -- kept as a
    # full copy (rather than importing and patching) so train.py stays
    # untouched and remains sigmoid-only.
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    loss_no_reduction = torch.nn.BCELoss(reduction='none')
    loss_history = []
    max_auc = 0

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')

    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')

    start_thresh = 1.0
    end_thresh = 0.0
    for epoch in range(args.epochs):
        t = time.time()
        if documentation:
            print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []
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

            weights = torch.ones_like(loss_simple)
            if curriculum:
                curv_flat = curv_batch.flatten()
                hard_mask = _mask_for(mode, current_threshold, curv_flat, temperature)
                weights = 1.0 + hard_weight * hard_mask
                loss_train = (loss_simple * weights).mean()

            if not curriculum:
                loss_train = loss_simple.mean()

            loss_history.append(loss_train)
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + log.flatten().tolist()

            if documentation:
                if i % 100 == 0:
                    print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                        loss_train.cpu().detach().numpy()) + ' current threshold:' + str(current_threshold))

        roc_train = roc_auc_score(y_label_train, y_pred_train)

        if not args.fastmode:
            roc_val, prc_val, f1_val, loss_val, _ = test(model, val_loader, data_o, args, use_curvature=use_curvature, weight_fct=weight_fct, final_train=False)
            if roc_val > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = roc_val

            if documentation:
                print('epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'auroc_train: {:.4f}'.format(roc_train),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'auroc_val: {:.4f}'.format(roc_val),
                      'auprc_val: {:.4f}'.format(prc_val),
                      'f1_val: {:.4f}'.format(f1_val),
                      'time: {:.4f}s'.format(time.time() - t))
        else:
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    auroc_test, prc_test, f1_test, loss_test, details = test(model_max, test_loader, data_o, args, use_curvature=use_curvature, weight_fct=weight_fct, final_train=True)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))
    if return_details:
        return auroc_test, details
    return auroc_test


# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

args = settings()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Load data
data_o, train_loader, val_loader, test_loader = load_data(args)

combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])

# K-Fold setup -- stratified so each fold matches the overall
# positive/negative ratio, rather than relying on shuffle alone
all_labels = np.concatenate([
    np.asarray(train_loader.dataset.label),
    np.asarray(val_loader.dataset.label)
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

HARD_WEIGHT = 4.0
TEMPERATURE = 0.2
MODES = ['hard', 'sigmoid', 'erf', 'ensemble']
scores = {mode: [] for mode in MODES}

# K-FOLD LOOP
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(combined_dataset)), all_labels)):

    print(f"\n========== FOLD {fold + 1}/5 ==========")

    # Extract POSITIVE TRAIN EDGES for graph
    train_edges = []
    for i in train_idx:
        label, (u, v), _ = combined_dataset[i]
        if label == 1:
            train_edges.append((u, v))

    # curvature for training edges, computed from this fold's training
    # graph alone (mirrors load_data()'s original train_cuva computation)
    edge_index_fold, edge_cuva_fold, edge_to_curv = get_curvature(
        unique_entity=data_o.num_nodes,
        positive=np.array(train_edges)
    )

    # Training loader -- curvature is freshly computed for this fold (see
    # build_fold_loader), not the stale curvature baked in at the single,
    # pre-fold load_data() call. Validation curvature is intentionally not
    # computed at all: test() only reads curv_batch when final_train=True,
    # which never happens for validation calls, so any value would be
    # unused -- and computing one from train+val edges would incorporate
    # held-out validation structure for no benefit.
    train_loader_fold = build_fold_loader(combined_dataset, train_idx, edge_to_curv, train_loader.batch_size, shuffle=True)
    val_loader_fold = DataLoader(Subset(combined_dataset, val_idx), batch_size=val_loader.batch_size, shuffle=False)

    # Prepare model data
    current_data = copy.deepcopy(data_o).to(device)
    current_data.edge_index = edge_index_fold.to(device)
    current_data.curva = edge_cuva_fold.to(device)

    for mode in MODES:
        print(f"Training {mode}...")

        model, opt = Create_model(args)

        if mode == 'sigmoid':
            auc = train_model(
                model, opt, current_data, train_loader_fold, val_loader_fold, test_loader, args,
                curriculum=True, hard_weight=HARD_WEIGHT, temperature=TEMPERATURE, use_curvature=False
            )
        else:
            auc = train_model_variant(
                mode, model, opt, current_data, train_loader_fold, val_loader_fold, test_loader, args,
                curriculum=True, hard_weight=HARD_WEIGHT, temperature=TEMPERATURE, use_curvature=False
            )

        scores[mode].append(auc)
        del model, opt

    # cleanup
    del current_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Results
print("\n" + "=" * 30)
for mode in MODES:
    s = scores[mode]
    print(f"{mode.upper()} MEAN AUC: {np.mean(s):.4f} ± {np.std(s):.4f}")
print("=" * 30)
