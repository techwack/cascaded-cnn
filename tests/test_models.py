from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from solarnet.common import augment, build_model, load_config  # noqa: E402
from solarnet.models import CascadedSolarNet  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/default.yaml")
TAX = Taxonomy.load(ROOT / "configs/taxonomy.yaml")


@pytest.mark.parametrize("node", TAX.nodes + ["flat"])
def test_node_output_shape(node):
    model = build_model(CFG, TAX, node).eval()
    for scale in (1, 2):  # native and 2x-upsampled input
        out = model(torch.zeros(4, 1, 40 * scale, 24 * scale))
        assert out.shape == (4, len(TAX.node_classes(node)))


def test_augment_keeps_shape_and_range():
    x = torch.rand(8, 1, 40, 24)
    y = augment(x, CFG["augment"])
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


class Fixed(torch.nn.Module):
    """Returns logits that pick column `x[:, 0, 0, col]` - lets a test script the routing."""

    def __init__(self, col: int, k: int):
        super().__init__()
        self.col, self.k = col, k

    def forward(self, x):
        return torch.nn.functional.one_hot(x[:, 0, 0, self.col].long(), self.k).float()


def test_cascade_routing_and_early_exit():
    fams = TAX.family_names
    offsets, s = {}, 1
    for f in fams:
        offsets[f] = s
        s += len(TAX.families[f])
    net = CascadedSolarNet(
        triage=Fixed(0, 2), family=Fixed(1, len(fams)),
        fine={f: Fixed(2, len(TAX.families[f])) for f in fams},
        family_names=fams, class_offsets=offsets,
    )
    # column 0: defect?, column 1: family, column 2: subtype
    x = torch.zeros(3, 1, 40, 24)
    x[1, 0, 0, :3] = torch.tensor([1, 0, 2])  # Electrical, 3rd class
    x[2, 0, 0, :3] = torch.tensor([1, 2, 1])  # Structural, 2nd class
    out = net.predict(x)
    assert out["pred"].tolist() == [0, TAX.index(TAX.families[fams[0]][2]), TAX.index(TAX.families[fams[2]][1])]
    assert out["is_defect"].tolist() == [False, True, True]
    assert out["family"].tolist() == [-1, 0, 2]
