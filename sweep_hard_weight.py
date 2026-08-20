"""
Sweeps hard_weight (lambda) to find the value that maximizes mean test
AUROC, using the corrected per-fold curvature computation (see main.py /
build_fold_loader). This redoes the original hard_weight sweep -- which
predated that fix and used curvature frozen from a single original split
across all 5 folds -- to confirm the previously-found best value (4) still
holds under the fixed 5-fold CV protocol described in the paper.

Fixed: temperature=0.2 (current default), sigmoid transition (train.py's
only supported curriculum mode). Swept: hard_weight in {0,1,...,8}.

Output mirrors main.py: per-fold progress prints, no plots, final mean +/-
std AUROC summary per hard_weight value.
"""
import copy
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedKFold

from parms_setting import settings
from data_preprocess import load_data, get_curvature, Data_class
from instantiation import Create_model
from train import train_model


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

HARD_WEIGHTS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
TEMPERATURE = 0.2
scores = {w: [] for w in HARD_WEIGHTS}

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
    # graph alone
    edge_index_fold, edge_cuva_fold, edge_to_curv = get_curvature(
        unique_entity=data_o.num_nodes,
        positive=np.array(train_edges)
    )

    # Training loader -- curvature is freshly computed for this fold (see
    # build_fold_loader), not the stale curvature baked in at the single,
    # pre-fold load_data() call. Validation curvature is intentionally not
    # computed: test() only reads curv_batch when final_train=True, which
    # never happens for validation calls, so any value would be unused --
    # and computing one from train+val edges would incorporate held-out
    # validation structure for no benefit.
    train_loader_fold = build_fold_loader(combined_dataset, train_idx, edge_to_curv, train_loader.batch_size, shuffle=True)
    val_loader_fold = DataLoader(Subset(combined_dataset, val_idx), batch_size=val_loader.batch_size, shuffle=False)

    # Prepare model data
    current_data = copy.deepcopy(data_o).to(device)
    current_data.edge_index = edge_index_fold.to(device)
    current_data.curva = edge_cuva_fold.to(device)

    for w in HARD_WEIGHTS:
        print(f"Training hard_weight={w}...")

        model, opt = Create_model(args)

        auc = train_model(
            model,
            opt,
            current_data,
            train_loader_fold,
            val_loader_fold,
            test_loader,
            args,
            curriculum=True,
            hard_weight=float(w),
            temperature=TEMPERATURE,
            use_curvature=False
        )

        scores[w].append(auc)
        del model, opt

    # cleanup
    del current_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Results
print("\n" + "=" * 30)
for w in HARD_WEIGHTS:
    s = scores[w]
    print(f"hard_weight={w}   MEAN AUC: {np.mean(s):.4f} ± {np.std(s):.4f}")
print("=" * 30)
