"""Write results/LIVE_STATUS.md from the training logs and copy finished result files into results/.

    python scripts/live_status.py
"""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def tail_epoch(log: Path, tag: str = ""):
    if not log.exists():
        return None
    lines = [l for l in log.read_text(encoding="utf-8", errors="ignore").splitlines() if re.search(r"ep +\d+ loss", l)]
    return lines[-1].strip()[:110] if lines else None


def main() -> None:
    out = [f"# Live status\n\nLast updated {datetime.now():%Y-%m-%d %H:%M} (written automatically by scripts/live_status.py).\n"]

    # pretrained cascade
    pr = ROOT / "runs_pretrained" / "results.json"
    out.append("## Pretrained cascade (EfficientNet-B0 under every stage)\n")
    if pr.exists():
        r = json.loads(pr.read_text())
        shutil.copy(pr, RES / "pretrained.json")
        for m in ("cascade", "flat"):
            if m in r:
                x = r[m]
                out.append(f"- **{m}**: accuracy {x['accuracy']:.4f}, macro-F1 {x['macro_f1']:.4f}, kappa {x['kappa']:.4f}")
        s = r["cascade"].get("stages")
        if s:
            out.append(f"- cascade stage errors: {s}")
    else:
        for name in ("pretrained_streamA.log", "pretrained_streamB.log"):
            e = tail_epoch(ROOT / name)
            if e:
                out.append(f"- {name}: `{e}`")
        out.append("- test evaluation not finished yet")

    # CV
    out.append("\n## 10-fold cross-validation of the pretrained cascade\n")
    folds = sorted((ROOT / "runs_cv").glob("fold*/metrics.json"))
    if folds:
        ms = [json.loads(p.read_text()) for p in folds]
        out.append("| fold | accuracy | macro-F1 | binary acc |\n|---|---|---|---|")
        for m in ms:
            out.append(f"| {m['fold']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['binary_accuracy']:.4f} |")
        acc = [m["accuracy"] for m in ms]
        out.append(f"\n{len(ms)}/10 folds done, mean accuracy so far {sum(acc) / len(acc):.4f}.")
        agg = RES / "cv_pretrained.json"
        # partial aggregate so teammates can see numbers before all folds finish
        keys = ("accuracy", "macro_f1", "kappa", "binary_accuracy")
        part = {"n_folds": len(ms), "folds": ms}
        for k in keys:
            v = [m[k] for m in ms]
            mean = sum(v) / len(v)
            part[k + "_mean"] = mean
            part[k + "_std"] = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5
        agg.write_text(json.dumps(part, indent=1))
    else:
        for name in ("cv_streamA.log", "cv_streamB.log"):
            e = tail_epoch(ROOT / name)
            if e:
                out.append(f"- {name}: `{e}`")
        if not any((ROOT / n).exists() for n in ("cv_streamA.log", "cv_streamB.log")):
            out.append("- queued (starts after the pretrained run)")
    (RES / "LIVE_STATUS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
