"""Define a catalytic design pocket around the whole bound complex or an inter-fragment interface.

Whole-complex connectivity logic:
- Admits non-polymer components (substrate, intermediate, cofactor, catalytic metals)
  by connectivity outward from an anchor within ``link_cutoff`` Å, preventing stray
  surface ions from importing spurious shell residues.
- Reports excluded components with their distance to the site rather than dropping
  them silently.

Interface logic:
- Identifies residues contacting two explicitly specified atom groups in the *same*
  pose across a structural ensemble, recording per-residue pose support counts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from attack_geometry.config import AtomSelector
from attack_geometry.structure import (
    Component,
    Structure,
    load_structure,
    min_distance,
    select_atoms,
)


@dataclass(frozen=True)
class Position:
    """One pocket residue and its distance relationship to the bound complex."""

    chain: str
    resnum: int
    resname: str
    wt_aa: str
    d_complex: float
    distances: dict[str, float]
    contacts: tuple[str, ...]

    @property
    def membership(self) -> str:
        return "+".join(self.contacts) if self.contacts else "none"


@dataclass(frozen=True)
class ExcludedComponent:
    component: str
    resname: str
    n_atoms: int
    d_to_site: float
    reason: str


@dataclass
class Pocket:
    structure_path: str
    cutoff: float
    link_cutoff: float
    anchor: str
    admitted: tuple[str, ...]
    excluded: tuple[ExcludedComponent, ...]
    positions: list[Position] = field(default_factory=list)

    def resnums(self, chain: str | None = None) -> list[int]:
        return [p.resnum for p in self.positions if chain is None or p.chain == chain]

    def to_rows(self) -> list[dict]:
        rows = []
        for p in self.positions:
            row = {
                "chain": p.chain,
                "resnum": p.resnum,
                "resname": p.resname,
                "wt_aa": p.wt_aa,
                "d_complex": round(p.d_complex, 3),
                "membership": p.membership,
                "n_contacts": len(p.contacts),
            }
            for cid in self.admitted:
                row[f"d_{cid}"] = round(p.distances.get(cid, float("inf")), 3)
            rows.append(row)
        return rows


def _select_components(structure: Structure, components: list[str] | None) -> list[Component]:
    if components is None:
        return list(structure.components)
    chosen, missing = [], []
    for cid in components:
        c = structure.component(cid)
        if c is None:
            missing.append(cid)
        elif c not in chosen:
            chosen.append(c)
    if missing:
        have = sorted({c.id for c in structure.components})
        raise ValueError(f"No such component(s) {missing} in {structure.path}; have {have}")
    return chosen


def admit_complex(
    comps: list[Component],
    link_cutoff: float = 5.0,
    anchor: str | None = None,
) -> tuple[list[Component], list[ExcludedComponent], str]:
    """Grow the active site outward from ``anchor`` by connectivity within ``link_cutoff`` Å."""
    if not comps:
        return [], [], ""
    if anchor is None:
        anchor_c = max(comps, key=lambda c: (c.n_atoms, c.id))
    else:
        matches = [c for c in comps if c.id == anchor or c.chain == anchor]
        if not matches:
            raise ValueError(
                f"Anchor {anchor!r} is not among selected components {[c.id for c in comps]}"
            )
        anchor_c = matches[0]

    admitted = [anchor_c]
    pending = [c for c in comps if c is not anchor_c]
    best: dict[str, float] = {c.id: float("inf") for c in pending}
    grew = True
    while grew:
        grew = False
        for c in list(pending):
            d = min(min_distance(c.xyz, a.xyz) for a in admitted)
            best[c.id] = min(best[c.id], d)
            if d <= link_cutoff:
                admitted.append(c)
                pending.remove(c)
                grew = True

    excluded = [
        ExcludedComponent(
            component=c.id,
            resname=c.resname,
            n_atoms=c.n_atoms,
            d_to_site=round(best[c.id], 3),
            reason=f"{best[c.id]:.2f} A from the site, beyond link_cutoff={link_cutoff} A",
        )
        for c in sorted(pending, key=lambda c: c.id)
    ]
    return admitted, excluded, anchor_c.id


def define_pocket(
    structure: str | Path | Structure,
    cutoff: float = 5.0,
    components: list[str] | None = None,
    anchor: str | None = None,
    link_cutoff: float = 5.0,
    require_connected: bool = True,
    protein_chain: str | None = None,
) -> Pocket:
    """Identify protein residues within ``cutoff`` Å of the connected bound complex."""
    st = load_structure(structure) if not isinstance(structure, Structure) else structure
    comps = _select_components(st, components)
    if not comps:
        raise ValueError(
            f"{st.path}: no non-polymer components found to define a pocket around."
        )

    if require_connected:
        admitted, excluded, anchor_id = admit_complex(comps, link_cutoff, anchor)
    else:
        admitted, excluded, anchor_id = comps, [], (anchor or comps[0].id)

    union = np.vstack([c.xyz for c in admitted])
    positions: list[Position] = []
    for r in st.residues:
        if protein_chain is not None and r.chain != protein_chain:
            continue
        d_complex = min_distance(r.xyz, union)
        if d_complex > cutoff:
            continue
        dists = {c.id: min_distance(r.xyz, c.xyz) for c in admitted}
        contacts = tuple(cid for cid in (c.id for c in admitted) if dists[cid] <= cutoff)
        positions.append(
            Position(
                chain=r.chain,
                resnum=r.resnum,
                resname=r.resname,
                wt_aa=r.aa1,
                d_complex=d_complex,
                distances=dists,
                contacts=contacts,
            )
        )

    positions.sort(key=lambda p: (p.chain, p.resnum))
    return Pocket(
        structure_path=st.path,
        cutoff=cutoff,
        link_cutoff=link_cutoff,
        anchor=anchor_id,
        admitted=tuple(c.id for c in admitted),
        excluded=tuple(excluded),
        positions=positions,
    )


def interface_positions(
    paths: Sequence[str | Path],
    protein_chain: str,
    fragment_a: AtomSelector | dict,
    fragment_b: AtomSelector | dict,
    cutoff: float = 5.0,
    min_contact_poses: int = 1,
) -> list[dict]:
    """Select residues contacting both ``fragment_a`` and ``fragment_b`` in the same pose."""
    path_objs = [Path(p).expanduser() for p in paths]
    if not path_objs or not 1 <= min_contact_poses <= len(path_objs) or cutoff <= 0:
        raise ValueError("Invalid ensemble size, min_contact_poses, or cutoff")
    if len({p.resolve() for p in path_objs}) != len(path_objs):
        raise ValueError("Duplicate ensemble paths supplied")

    sel_a = fragment_a if isinstance(fragment_a, AtomSelector) else AtomSelector.from_dict(fragment_a)
    sel_b = fragment_b if isinstance(fragment_b, AtomSelector) else AtomSelector.from_dict(fragment_b)

    counts: Counter[int] = Counter()
    distances: dict[int, list[tuple[float, float]]] = {}
    sequence: str | None = None

    for path in path_objs:
        st = load_structure(path)
        seq = st.sequence(protein_chain)
        if sequence is not None and seq != sequence:
            raise ValueError("Wild-type ensemble sequences differ across poses")
        sequence = seq
        coords_a = np.asarray([a.xyz for a in select_atoms(st, sel_a)], dtype=float)
        coords_b = np.asarray([a.xyz for a in select_atoms(st, sel_b)], dtype=float)
        for r in st.residues:
            if r.chain != protein_chain:
                continue
            da = min_distance(r.xyz, coords_a)
            db = min_distance(r.xyz, coords_b)
            if da <= cutoff and db <= cutoff:
                counts[r.resnum] += 1
                distances.setdefault(r.resnum, []).append((da, db))

    rows = []
    assert sequence is not None
    for rn, n in sorted(counts.items()):
        if n < min_contact_poses:
            continue
        rows.append(
            {
                "chain": protein_chain,
                "resnum": rn,
                "wt_aa": sequence[rn - 1],
                "n_contact_poses": n,
                "n_poses": len(path_objs),
                "contact_fraction": round(n / len(path_objs), 4),
                "min_joint_max_distance_A": round(
                    min(max(da, db) for da, db in distances[rn]), 3
                ),
            }
        )
    return rows
