"""Random Forest on raw pixels: the classical baseline from the original benchmark.

    python baselines/random_forest.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score  # noqa: E402

from solarnet.data import load_splits  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tax = Taxonomy.load(args.taxonomy)
    splits, _ = load_splits(cfg, tax)

    def xy(split):
        d = splits[split]
        return d.images.reshape(len(d.images), -1), d.global_targets(tax)

    x_tr, y_tr = xy("train")
    x_te, y_te = xy("test")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                                random_state=cfg["data"]["seed"]).fit(x_tr, y_tr)
    pred = rf.predict(x_te)
    res = {"accuracy": accuracy_score(y_te, pred),
           "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
           "kappa": cohen_kappa_score(y_te, pred)}
    print(json.dumps(res, indent=1))
    out = Path(cfg["train"]["out_dir"]) / "random_forest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
