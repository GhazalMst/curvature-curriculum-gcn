import torch
import numpy as np
import copy
import networkx as nx
from torch.utils.data import Subset, DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedKFold
from parms_setting import settings
from data_preprocess import load_data, get_curvature, get_proxy_curvature, Data_class
from instantiation import Create_model
from train import train_model
from GraphRicciCurvature.OllivierRicci import OllivierRicci


def build_fold_loader(combined_dataset, idx, edge_to_curv, fold_positive_edges, unique_entity, batch_size, shuffle):
    # Rebuilds (u, v, label) triples plus a *freshly computed* curvature
    # value per sample for this fold, instead of reusing the curvature
    # baked into combined_dataset at the single, pre-fold load_data() call --
    # that stale curvature doesn't respect the current fold's train/val
    # boundary, since it was computed once from a different, original split.

    # nodes with zero edges in THIS fold's training graph -- a negative pair
    # touching one has no real neighborhood for get_proxy_curvature to work
    # from, so it always falls back to DISCONNECTED_CURVATURE. Positive edges
    # can never touch these nodes (zero degree here means it can't be a
    # positive-edge endpoint in this fold), so this only ever drops negatives.
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
        label, (u, v), _ = combined_dataset[i]
        u, v, label = int(u), int(v), int(label)
        if label == 0 and (isolated_this_fold[u] or isolated_this_fold[v]):
            # drop rather than train on a fake "confidently easy" value
            continue
        triples.append((u, v, label))
        if label == 1:
            # self-loops (a node interacting with itself) get curvature 0 by
            # convention, same as get_curvature() -- they're deliberately not
            # stored in edge_to_curv, so looking one up there would KeyError
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

combined_dataset = ConcatDataset([
    train_loader.dataset,
    val_loader.dataset
])


# K-Fold setup -- stratified so each fold matches the overall
# positive/negative ratio, rather than relying on shuffle alone
all_labels = np.concatenate([
    np.asarray(train_loader.dataset.label),
    np.asarray(val_loader.dataset.label)
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

baseline_scores = {'auroc': [], 'auprc': [], 'f1': [], 'acc': [], 'loss': []}
curriculum_scores = {'auroc': [], 'auprc': [], 'f1': [], 'acc': [], 'loss': []}

def _record(store, details):
    store['auroc'].append(details['auroc_test'])
    store['auprc'].append(details['prc_test'])
    store['f1'].append(details['f1_test'])
    store['acc'].append(details['acc_test'])
    store['loss'].append(details['loss_test'])

def _sem(values):
    return np.std(values, ddof=1) / np.sqrt(len(values))


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
    train_loader_fold = build_fold_loader(combined_dataset, train_idx, edge_to_curv, np.array(train_edges), data_o.num_nodes, train_loader.batch_size, shuffle=True)
    val_loader_fold = DataLoader(Subset(combined_dataset, val_idx), batch_size=val_loader.batch_size, shuffle=False)

    # Prepare model data
    current_data = copy.deepcopy(data_o).to(device)

    current_data.edge_index = edge_index_fold.to(device)
    current_data.curva = edge_cuva_fold.to(device)


    # Train baseline
    print("Training Baseline...")

    model_b, opt_b = Create_model(args)

    auc_b, details_b = train_model(
        model_b,
        opt_b,
        current_data,
        train_loader_fold,
        val_loader_fold,
        test_loader,
        args,
        curriculum=False,
        use_curvature=False,
        return_details=True
    )

    _record(baseline_scores, details_b)


    # Train curriculum
    print("Training Curriculum...")

    model_c, opt_c = Create_model(args)

    auc_c, details_c = train_model(
        model_c,
        opt_c,
        current_data,
        train_loader_fold,
        val_loader_fold,
        test_loader,
        args,
        curriculum=True,
        use_curvature=False,
        return_details=True
    )

    _record(curriculum_scores, details_c)


    # cleanup
    del model_b, opt_b, model_c, opt_c, current_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Results
print("\n" + "=" * 30)
for name, scores in [("BASELINE", baseline_scores), ("CURRICULUM", curriculum_scores)]:
    print(f"{name} MEAN AUC:   {np.mean(scores['auroc']):.4f} ± {_sem(scores['auroc']):.4f}")
    print(f"{name} MEAN AUPRC: {np.mean(scores['auprc']):.4f} ± {_sem(scores['auprc']):.4f}")
    print(f"{name} MEAN F1:    {np.mean(scores['f1']):.4f} ± {_sem(scores['f1']):.4f}")
    print(f"{name} MEAN ACC:   {np.mean(scores['acc']):.4f} ± {_sem(scores['acc']):.4f}")
    print(f"{name} MEAN LOSS:  {np.mean(scores['loss']):.4f} ± {_sem(scores['loss']):.4f}")
print("=" * 30)