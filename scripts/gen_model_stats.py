"""Layer-by-layer shapes and parameter counts of the stage networks -> results/model_stats.json (run with the project venv)."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarnet.common import build_model, load_config  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402

tax = Taxonomy.load(ROOT / "configs" / "taxonomy.yaml")
out = {}
for cfg_name, in_shape in (("improved3", (1, 1, 80, 48)), ("pretrained", (1, 1, 40, 24))):
    cfg = load_config(ROOT / "configs" / f"{cfg_name}.yaml")
    nodes = {}
    for node in tax.nodes:
        m = build_model(cfg, tax, node).eval()
        rows = []
        if cfg_name == "improved3":
            x = torch.zeros(in_shape)
            for i, layer in enumerate(m.features):
                x = layer(x)
                p = sum(q.numel() for q in layer.parameters())
                rows.append([f"{type(layer).__name__}" + (f"({layer.out_channels})" if isinstance(layer, nn.Conv2d) else ""), list(x.shape[1:]), p])
            g = x.mean((2, 3))
            rows.append(["GlobalAvgPool", list(g.shape[1:]), 0])
            fc = [q for q in m.head if isinstance(q, nn.Linear)][0]
            rows.append([f"FC({fc.out_features})", [fc.out_features], sum(q.numel() for q in fc.parameters())])
        nodes[node] = {"params": sum(q.numel() for q in m.parameters()), "layers": rows}
    out[cfg_name] = nodes
(ROOT / "results" / "model_stats.json").write_text(json.dumps(out, indent=1))
print({c: {n: v["params"] for n, v in d.items()} for c, d in out.items()})
