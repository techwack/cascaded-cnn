import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from solarnet.data import class_weights, load_records, make_splits
from solarnet.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tax():
    return Taxonomy.load(ROOT / "configs/taxonomy.yaml")


def test_twelve_classes_healthy_first(tax):
    assert len(tax.classes) == 12
    assert tax.classes[0] == tax.healthy
    assert tax.nodes == ["triage", "family", "electrical", "environmental", "structural"]


def test_node_labels_cover_every_class_exactly_once(tax):
    for cls in tax.classes[1:]:
        fam = tax.family_of(cls)
        assert tax.node_label("triage", cls) == 1
        assert tax.node_label("family", cls) == tax.family_names.index(fam)
        seen = [n for n in tax.nodes[2:] if tax.node_label(n, cls) is not None]
        assert seen == [fam.lower()]
    assert tax.node_label("triage", tax.healthy) == 0
    assert all(tax.node_label(n, tax.healthy) is None for n in tax.nodes[1:])


def test_global_index_matches_family_offsets(tax):
    offset = 1
    for fam, cs in tax.families.items():
        for i, c in enumerate(cs):
            assert tax.index(c) == offset + i == offset + tax.node_label(fam.lower(), c)
        offset += len(cs)


def test_duplicate_class_rejected(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("healthy: H\nfamilies:\n  A: [x, y]\n  B: [y, z]\n")
    with pytest.raises(ValueError, match="more than once"):
        Taxonomy.load(p)


def test_class_weights_formula():
    y = np.array([0] * 6 + [1] * 3 + [2])
    w = class_weights(y, 4)
    np.testing.assert_allclose(w, [10 / (4 * 6), 10 / (4 * 3), 10 / 4, 0.0])


def fake_dataset(root: Path, tax: Taxonomy, per_class: int = 20) -> None:
    (root / "images").mkdir(parents=True)
    meta, rng, k = {}, np.random.default_rng(0), 0
    for cls in tax.classes:
        for _ in range(per_class):
            Image.fromarray(rng.integers(0, 255, (40, 24), dtype=np.uint8)).save(root / f"images/{k}.jpg")
            meta[str(k)] = {"image_filepath": f"images/{k}.jpg", "anomaly_class": cls}
            k += 1
    (root / "module_metadata.json").write_text(json.dumps(meta))


def test_stratified_split_is_disjoint_and_reproducible(tmp_path, tax):
    fake_dataset(tmp_path, tax)
    recs = load_records(tmp_path)
    a = make_splits(recs, 0.15, 0.15, seed=42)
    b = make_splits(recs, 0.15, 0.15, seed=42)
    assert a == b
    ids = a["train"] + a["val"] + a["test"]
    assert len(ids) == len(set(ids)) == len(recs)
    label = {r["id"]: r["label"] for r in recs}
    for split in ("val", "test"):  # every class present in every split
        assert set(Counter(label[i] for i in a[split])) == set(tax.classes)
