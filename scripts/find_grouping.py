"""Find the 3-way family grouping that keeps confusable classes together.

Uses the trained flat CNN's confusion matrix on the VALIDATION split (never the test
split), then brute-forces every partition of the fault classes into K families
(each with >= 2 classes) and keeps the one with the fewest cross-family confusions.

    python scripts/find_grouping.py --config configs/improved.yaml --flat-dir runs_improved

Writes configs/taxonomy_data.yaml and prints the cross-family confusions for it and for
the current taxonomy.yaml, so the two groupings can be compared.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solarnet.common import load_config, load_node, preprocess, to_batch  # noqa: E402
from solarnet.data import load_splits  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402


def partitions(items, k):
    """All ways to split `items` into k non-empty unlabeled groups."""
    if not items:
        if k == 0:
            yield []
        return
    if k == 0:
        return
    first, rest = items[0], items[1:]
    for p in partitions(rest, k):  # first joins an existing group
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
    for p in partitions(rest, k - 1):  # first starts a new group
        yield [[first]] + p


def cross(conf, groups):
    sym = conf + conf.T
    total = 0
    for a, b in itertools.combinations(range(len(groups)), 2):
        total += sum(sym[i, j] for i in groups[a] for j in groups[b])
    return int(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/improved.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--flat-dir", default="runs_improved")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--out", default="configs/taxonomy_data.yaml")
    args = ap.parse_args()

    cfg, tax = load_config(args.config), Taxonomy.load(args.taxonomy)
    splits, stats = load_splits(cfg, tax)
    val = splits["val"]
    x = preprocess(to_batch(val.images), stats, cfg["data"]["upsample"])
    y = val.global_targets(tax)
    flat = load_node(cfg, tax, "flat", Path(args.flat_dir))
    with torch.no_grad():
        pred = torch.cat([flat(x[i:i + 250]).argmax(1) for i in range(0, len(x), 250)]).numpy()

    n = len(tax.classes)
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    np.fill_diagonal(conf, 0)  # only the mistakes matter
    faults = list(range(1, n))  # index 0 is the healthy class

    current = [[tax.index(c) for c in cs] for cs in tax.families.values()]
    print(f"current taxonomy: cross-family confusions on val = {cross(conf, current)}")

    best = min((p for p in partitions(faults, args.k) if all(len(g) >= args.min_size for g in p)),
               key=lambda p: cross(conf, p))
    best = sorted((sorted(g) for g in best), key=lambda g: g[0])
    print(f"best data-driven grouping: cross-family confusions on val = {cross(conf, best)}")
    names = ["GroupA", "GroupB", "GroupC"][:args.k]
    out = {"healthy": tax.healthy, "families": {nm: [tax.classes[i] for i in g] for nm, g in zip(names, best)}}
    for nm, cs in out["families"].items():
        print(f"  {nm}: {cs}")
    Path(args.out).write_text("# Data-driven families (see scripts/find_grouping.py)\n" + yaml.safe_dump(out, sort_keys=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
