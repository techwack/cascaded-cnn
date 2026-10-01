"""Class hierarchy and per-node label views.

The cascade has five nodes:
    triage      CNN-1  healthy (0) vs defect (1)
    family      CNN-2  index into Taxonomy.families
    <family>    CNN-3x one per family, index into that family's class list
plus `flat`, the single-stage 12-class baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Taxonomy:
    healthy: str
    families: dict[str, list[str]]

    @classmethod
    def load(cls, path: str | Path) -> "Taxonomy":
        with open(path) as f:
            raw = yaml.safe_load(f)
        tax = cls(healthy=raw["healthy"], families={k: list(v) for k, v in raw["families"].items()})
        tax._validate()
        return tax

    def _validate(self) -> None:
        seen = [self.healthy] + [c for cs in self.families.values() for c in cs]
        dupes = {c for c in seen if seen.count(c) > 1}
        if dupes:
            raise ValueError(f"classes listed more than once: {sorted(dupes)}")
        for fam, cs in self.families.items():
            if len(cs) < 2:
                raise ValueError(f"family {fam!r} needs at least 2 classes for its own CNN")

    @property
    def family_names(self) -> list[str]:
        return list(self.families)

    @property
    def classes(self) -> list[str]:
        """Global class order: healthy first, then families in order."""
        return [self.healthy] + [c for cs in self.families.values() for c in cs]

    @property
    def nodes(self) -> list[str]:
        return ["triage", "family", *[node_name(f) for f in self.families]]

    def index(self, cls_name: str) -> int:
        return self.classes.index(cls_name)

    def family_of(self, cls_name: str) -> str | None:
        for fam, cs in self.families.items():
            if cls_name in cs:
                return fam
        return None

    def node_classes(self, node: str) -> list[str]:
        if node == "triage":
            return [self.healthy, "Defect"]
        if node == "family":
            return self.family_names
        if node == "flat":
            return self.classes
        return self.families[self._family_for_node(node)]

    def node_label(self, node: str, cls_name: str) -> int | None:
        """Target index for `cls_name` at `node`, or None if the node never sees it."""
        if node == "flat":
            return self.index(cls_name)
        if node == "triage":
            return 0 if cls_name == self.healthy else 1
        fam = self.family_of(cls_name)
        if fam is None:
            return None
        if node == "family":
            return self.family_names.index(fam)
        if fam != self._family_for_node(node):
            return None
        return self.families[fam].index(cls_name)

    def _family_for_node(self, node: str) -> str:
        for fam in self.families:
            if node_name(fam) == node:
                return fam
        raise KeyError(f"unknown node {node!r}; expected one of {self.nodes + ['flat']}")


def node_name(family: str) -> str:
    return family.lower()
