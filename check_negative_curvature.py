"""Prints the interval of proxy curvature for negative (non-edge) pairs,
separately from positive edges and separately from the disconnected-pair
floor (DISCONNECTED_CURVATURE), for whatever dataset/seed args.in_file /
args.seed point to.

Usage: python check_negative_curvature.py [--in_file data/HuRI2.edgelist] [--seed 5327] ...
(accepts the same CLI args as main.py, via parms_setting.settings())
"""
import torch
from parms_setting import settings
from data_preprocess import load_data, DISCONNECTED_CURVATURE


def report(name, curv):
    if len(curv) == 0:
        print(f"  {name}: no samples")
        return
    print(f"  {name}: n={len(curv)}  min={curv.min():.4f}  max={curv.max():.4f}  "
          f"mean={curv.mean():.4f}  std={curv.std():.4f}")
    pct = "  ".join(f"p{int(q*100)}={torch.quantile(curv, q):.4f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95))
    print(f"    percentiles: {pct}")


def summarize_split(loader, split_name):
    label = torch.as_tensor(loader.dataset.label)
    curv = loader.dataset.curvature

    pos_curv = curv[label == 1]
    neg_curv = curv[label == 0]

    finite_neg = neg_curv[torch.isfinite(neg_curv)]
    inf_count = (~torch.isfinite(neg_curv)).sum().item()
    disconnected = finite_neg[finite_neg == DISCONNECTED_CURVATURE]
    real_neg = finite_neg[finite_neg > DISCONNECTED_CURVATURE]

    print(f"\n=== {split_name} ===")
    print(f"  negatives: {len(neg_curv)} total  |  "
          f"{len(disconnected)} disconnected (pinned at {DISCONNECTED_CURVATURE})  |  "
          f"{inf_count} inf (not computed by this loader)  |  "
          f"{len(real_neg)} real/connected")
    report("negatives (real/connected only, excludes disconnected floor and inf)", real_neg)
    report("positives (for comparison)", pos_curv)


if __name__ == "__main__":
    args = settings()
    args.cuda = torch.cuda.is_available()
    data_o, train_loader, val_loader, test_loader = load_data(args)

    summarize_split(train_loader, "Train")
    summarize_split(val_loader, "Val")
    summarize_split(test_loader, "Test")
