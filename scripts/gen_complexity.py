"""Count multiply-accumulates (convolution + linear layers) of each stage network -> results/complexity.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarnet.common import load_cascade, load_config, preprocess, to_batch
from solarnet.taxonomy import Taxonomy

tax = Taxonomy.load(ROOT / "configs" / "taxonomy.yaml")

def macs(net, x):
    tot = [0]
    def hook(m, i, o):
        if isinstance(m, torch.nn.Conv2d):
            tot[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, torch.nn.Linear):
            tot[0] += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        net.eval()(x)
    for h in hs:
        h.remove()
    return tot[0] / 1e9

out = {}
for name in ("improved3", "pretrained"):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    saved = json.load(open(cfg["data"]["splits_file"]))
    casc = load_cascade(cfg, tax, ROOT / cfg["train"]["out_dir"]).eval()
    x = preprocess(to_batch(np.zeros((1, 40, 24), np.float32)), saved["stats"], cfg["data"].get("upsample", 2) if name != "pretrained" else 1) if False else None
    img = np.random.rand(40, 24).astype(np.float32)
    x = preprocess(to_batch(img[None]), saved["stats"], 2 if name == "improved3" else 1)
    out[name] = {"triage": macs(casc.triage, x), "family": macs(casc.family, x), **{f"fine_{k}": macs(v, x) for k, v in casc.fine.items()}, "input": list(x.shape)}
    print(name, out[name])
json.dump(out, open(ROOT / "results" / "complexity.json", "w"), indent=1)
