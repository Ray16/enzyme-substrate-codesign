"""Versioned, hydrogen-free protein-donor / ligand-acceptor contact audit (v2).

Extracted verbatim in scientific policy from ``enzyme_design_agent/contact_analysis.py``.
Identifies chemically compatible heavy-atom contacts (not proven hydrogen bonds)
with explicit lower-bound anomalous-short quarantine (< 2.4 Å), covalent-radius
severe clash screening, conditional His/Cys separation, and atom-disjoint
two-point matching.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ContactPolicy:
    version: str = "heavy_atom_donor_acceptor_v2"
    minimum_contact_A: float = 2.4
    polar_max_A: float = 3.5
    ionic_max_A: float = 4.0
    audit_max_A: float = 4.5
    scope: str = "protein_sidechain_to_ligand_acceptor"

    def __post_init__(self) -> None:
        if self.scope != "protein_sidechain_to_ligand_acceptor":
            raise ValueError(
                "Unsupported scope; backbone and other interaction classes need explicit profiles"
            )
        if not 0 < self.polar_max_A <= self.audit_max_A or not 0 < self.ionic_max_A <= self.audit_max_A:
            raise ValueError("Invalid distance thresholds")
        if not 0 < self.minimum_contact_A < min(self.polar_max_A, self.ionic_max_A):
            raise ValueError("Invalid minimum contact distance")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Atom:
    chain: str
    residue: str
    resi: str
    name: str
    element: str
    xyz: tuple[float, float, float]
    confidence: float | None = None

    def __post_init__(self) -> None:
        if len(self.xyz) != 3 or not all(math.isfinite(v) for v in self.xyz):
            raise ValueError("Coordinates must be three finite numbers")
        if not self.chain or not self.resi or not self.name:
            raise ValueError("Complete atom identity is required")


@dataclass(frozen=True)
class TargetGroup:
    """Named acceptor atoms and formal charge of a resonance-delocalized ligand group."""

    group_id: str
    chain: str
    resi: str
    acceptors: tuple[str, ...]
    formal_charge: int
    provenance: str


DONORS: dict[str, set[str]] = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "TRP": {"NE1"},
    "HIS": {"ND1", "NE2"},
    "CYS": {"SG"},
}

STANDARD = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
CATIONIC: dict[str, set[str]] = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}}
COVALENT_RADII: dict[str, float] = {"N": 0.71, "O": 0.66, "S": 1.05}


def group_key(c: dict) -> tuple[str, str, str, str, str, str]:
    return (c["chain"], c["ri"], c["res"], c["target_chain"], c["target_resi"], c["group"])


def evaluate_contacts(
    atoms: Iterable[Atom],
    groups: Iterable[TargetGroup],
    policy: ContactPolicy = ContactPolicy(),
    protein_chain: str = "A",
) -> list[dict]:
    """Audit all nearby sidechain N/O/S pairs, retaining explicit rejection reasons."""
    atom_list, group_list = list(atoms), list(groups)
    amap = {(a.chain, a.resi, a.name): a for a in atom_list}
    if len(amap) != len(atom_list):
        raise ValueError("Duplicate atom identities / unresolved alternate conformations")

    prot = [
        a
        for a in atom_list
        if a.chain == protein_chain
        and a.element in {"N", "O", "S"}
        and a.name not in {"N", "O", "OXT"}
    ]
    out: list[dict] = []
    for g in group_list:
        if not g.acceptors or not g.provenance:
            raise ValueError("Acceptor identity and chemical provenance are required")
        if len(set(g.acceptors)) != len(g.acceptors):
            raise ValueError("Duplicate acceptor identities")
        for name in g.acceptors:
            target = amap[(g.chain, g.resi, name)]
            if target.element not in {"N", "O", "S"}:
                raise ValueError("Unsupported acceptor element for this profile")
            for a in prot:
                d = float(np.linalg.norm(np.subtract(a.xyz, target.xyz)))
                if d > policy.audit_max_A:
                    continue
                donor = a.name in DONORS.get(a.residue, set())
                ionic = a.name in CATIONIC.get(a.residue, set()) and g.formal_charge < 0
                kind = "potential_ionic" if ionic else "donor_acceptor_polar"
                limit = policy.ionic_max_A if ionic else policy.polar_max_A
                ambiguous = a.residue in {"HIS", "CYS"}
                severe_clash = d < COVALENT_RADII.get(a.element, 0.7) + COVALENT_RADII.get(
                    target.element, 0.7
                )
                if a.residue not in STANDARD:
                    reason = "untyped_residue"
                elif not donor:
                    reason = "incompatible_donor_acceptor"
                elif severe_clash:
                    reason = "severe_clash"
                elif d < policy.minimum_contact_A:
                    reason = "anomalously_short_contact"
                elif d > limit:
                    reason = "beyond_class_distance"
                elif ambiguous:
                    reason = "conditional_protonation_or_tautomer"
                else:
                    reason = "compatible_heavy_atom_contact"
                out.append(
                    {
                        "chain": a.chain,
                        "res": a.residue,
                        "ri": a.resi,
                        "atom": a.name,
                        "target_chain": g.chain,
                        "target_resi": g.resi,
                        "acceptor_atom": name,
                        "group": g.group_id,
                        "d": round(d, 4),
                        "kind": kind,
                        "cutoff_A": limit,
                        "accepted": reason == "compatible_heavy_atom_contact",
                        "conditional": reason
                        in {"conditional_protonation_or_tautomer", "anomalously_short_contact"},
                        "reason": reason,
                    }
                )

    flagged = {
        group_key(c) for c in out if c["reason"] in {"severe_clash", "anomalously_short_contact"}
    }
    for c in out:
        if group_key(c) in flagged and (c["accepted"] or c["conditional"]):
            c["accepted"] = False
            c["conditional"] = True
            if c["reason"] not in {"severe_clash", "anomalously_short_contact"}:
                c["reason"] = "group_contains_anomalously_short_contact"
    return out


def collapse_for_display(contacts: list[dict]) -> list[dict]:
    """One displayed line per residue/target group while retaining maximum atom-disjoint matching."""
    grouped: dict[tuple, list[dict]] = {}
    for c in contacts:
        if c["accepted"] or c["conditional"]:
            grouped.setdefault(group_key(c), []).append(c)
    result: list[dict] = []
    for _key, pairs in sorted(grouped.items()):
        pairs = sorted(pairs, key=lambda c: (c["d"], c["atom"], c["acceptor_atom"]))
        best = [pairs[0]]
        max_size = min(len({p["atom"] for p in pairs}), len({p["acceptor_atom"] for p in pairs}))
        for size in range(2, max_size + 1):
            choices = [
                s
                for s in combinations(pairs, size)
                if len({p["atom"] for p in s}) == size
                and len({p["acceptor_atom"] for p in s}) == size
            ]
            if choices:
                best = list(min(choices, key=lambda s: sum(p["d"] for p in s)))
        result.append(
            {
                **{
                    k: pairs[0][k]
                    for k in ("chain", "res", "ri", "target_chain", "target_resi", "group", "kind", "conditional")
                },
                "pairs": [pairs[0]],
                "independent_pairs": best,
                "uncertainty_reasons": sorted({p["reason"] for p in pairs if p["conditional"]}),
                "atom_pair_count": len(pairs),
                "display_line_count": 1,
                "collapsed_atom_pairs": len(pairs) - 1,
            }
        )
    return result


def contact_keys(contacts: list[dict]) -> set[tuple]:
    return {group_key(c) for c in contacts if c["accepted"]}


def distinct_residue_count(contacts: list[dict]) -> int:
    return len({(c["chain"], c["ri"]) for c in contacts if c["accepted"]})
