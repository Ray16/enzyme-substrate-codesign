"""Catalytic attack-geometry evaluation, ensemble aggregation, and motif alignment.

Preserves the exact nucleophile-electrophile distance, attack angle/trajectory,
proximity-weighted ensemble summarization, bootstrap WT comparison, Kabsch
superposition with ASP OD1/OD2 symmetry resolution, and protein-ligand steric
clash metrics from ``enzyme_design_agent``.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import GeometryConfig, ThresholdConfig
from .structure import Structure, load_structure, select_atoms


@dataclass(frozen=True)
class AttackGeometry:
    """Per-pose nucleophile-electrophile attack geometry."""

    distance_A: float
    angle_deg: float
    attack_chain: str
    attack_resnum: int
    attack_atom: str
    center_chain: str
    center_atom: str
    leaving_chain: str
    leaving_atom: str
    n_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance_A": round(self.distance_A, 4),
            "angle_deg": round(self.angle_deg, 3),
            "attack_chain": self.attack_chain,
            "attack_resnum": self.attack_resnum,
            "attack_atom": self.attack_atom,
            "center_chain": self.center_chain,
            "center_atom": self.center_atom,
            "leaving_chain": self.leaving_chain,
            "leaving_atom": self.leaving_atom,
            "n_candidates": self.n_candidates,
        }


@dataclass(frozen=True)
class EnsembleGeometry:
    """Proximity-weighted ensemble attack geometry summary."""

    n_poses: int
    weighted_angle_deg: float
    harmonic_distance_A: float
    mean_distance_A: float
    median_distance_A: float
    mean_angle_deg: float
    median_angle_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_poses": self.n_poses,
            "weighted_angle_deg": round(self.weighted_angle_deg, 3),
            "harmonic_distance_A": round(self.harmonic_distance_A, 4),
            "mean_distance_A": round(self.mean_distance_A, 4),
            "median_distance_A": round(self.median_distance_A, 4),
            "mean_angle_deg": round(self.mean_angle_deg, 3),
            "median_angle_deg": round(self.median_angle_deg, 3),
        }


def evaluate_attack_geometry(
    structure: str | Structure,
    config: GeometryConfig,
) -> AttackGeometry:
    """Compute nucleophile-electrophile distance and attack angle (degrees).

    The attack angle is the angle ``donor_best -- electrophile -- leaving``,
    where ``donor_best`` is the atom in ``config.donor`` closest to
    ``config.electrophile`` (ties broken deterministically by atom order).
    """
    st = load_structure(structure) if not isinstance(structure, Structure) else structure
    donors = select_atoms(st, config.donor)
    centers = select_atoms(st, config.electrophile)
    leavings = select_atoms(st, config.leaving)
    if len(centers) != 1 or len(leavings) != 1:
        raise ValueError(
            f"{st.path}: electrophile ({len(centers)}) and leaving ({len(leavings)}) "
            "selectors must each identify exactly one atom."
        )

    center = centers[0]
    leaving = leavings[0]
    donor_coords = np.asarray([a.xyz for a in donors], dtype=float)
    if not (
        np.isfinite(donor_coords).all()
        and np.isfinite(center.xyz).all()
        and np.isfinite(leaving.xyz).all()
    ):
        raise ValueError(f"{st.path}: non-finite coordinates in reactive atoms")

    distances = np.linalg.norm(donor_coords - center.xyz, axis=1)
    best_idx = int(np.argmin(distances))
    best_donor = donors[best_idx]
    d = float(distances[best_idx])

    u = donor_coords[best_idx] - center.xyz
    v = leaving.xyz - center.xyz
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= 0.0:
        raise ValueError(f"{st.path}: degenerate reactive geometry (zero vector norm)")
    cos_theta = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    angle = math.degrees(math.acos(cos_theta))

    return AttackGeometry(
        distance_A=d,
        angle_deg=angle,
        attack_chain=best_donor.chain,
        attack_resnum=best_donor.resnum,
        attack_atom=best_donor.name,
        center_chain=center.chain,
        center_atom=center.name,
        leaving_chain=leaving.chain,
        leaving_atom=leaving.name,
        n_candidates=len(donors),
    )


def proximity_weighted_angle_and_distance(poses: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return ``(..., 2)`` array of ``(weighted_angle_deg, harmonic_distance_A)``.

    Input ``poses`` has shape ``(..., n_poses, 2)`` where the final axis is
    ``(distance_A, angle_deg)``. Proximity weight is ``w_i = 1 / d_i``:
    - ``weighted_angle = sum(angle_i / d_i) / sum(1 / d_i)``
    - ``harmonic_distance = N / sum(1 / d_i)``
    """
    x = np.asarray(poses, dtype=float)
    if x.ndim < 2 or x.shape[-1] != 2 or x.shape[-2] == 0:
        raise ValueError("Expected a nonempty pose axis and a (distance, angle) final axis")
    if np.any(~np.isfinite(x)) or np.any(x[..., 0] <= 0):
        raise ValueError("Expected finite positive distances and finite angles")
    if np.any((x[..., 1] < 0) | (x[..., 1] > 180)):
        raise ValueError("Angles must be in degrees between 0 and 180")
    d, a = x[..., 0], x[..., 1]
    w = 1.0 / d
    w_sum = w.sum(axis=-1)
    return np.stack([(a * w).sum(axis=-1) / w_sum, d.shape[-1] / w_sum], axis=-1)


def summarize_ensemble(geometries: Sequence[AttackGeometry]) -> EnsembleGeometry:
    """Aggregate per-pose attack geometries into proximity-weighted ensemble metrics."""
    if not geometries:
        raise ValueError("Cannot summarize an empty pose ensemble")
    arr = np.array([[g.distance_A, g.angle_deg] for g in geometries], dtype=float)
    weighted = proximity_weighted_angle_and_distance(arr)
    return EnsembleGeometry(
        n_poses=len(geometries),
        weighted_angle_deg=float(weighted[0]),
        harmonic_distance_A=float(weighted[1]),
        mean_distance_A=float(np.mean(arr[:, 0])),
        median_distance_A=float(np.median(arr[:, 0])),
        mean_angle_deg=float(np.mean(arr[:, 1])),
        median_angle_deg=float(np.median(arr[:, 1])),
    )


def bootstrap_wt_delta(
    variant_poses: np.ndarray,
    wt_poses: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Independent ensemble bootstrap comparison against matched wild-type poses."""
    var_summary = proximity_weighted_angle_and_distance(variant_poses)
    wt_summary = proximity_weighted_angle_and_distance(wt_poses)
    delta = var_summary - wt_summary
    rng = np.random.default_rng(seed)
    bs_var = variant_poses[rng.integers(len(variant_poses), size=(n_bootstrap, len(variant_poses)))]
    bs_wt = wt_poses[rng.integers(len(wt_poses), size=(n_bootstrap, len(wt_poses)))]
    bs = (
        proximity_weighted_angle_and_distance(bs_var)
        - proximity_weighted_angle_and_distance(bs_wt)
    )
    ci = np.percentile(bs, [5, 95], axis=0)
    return {
        "delta_angle_deg": round(float(delta[0]), 4),
        "delta_distance_A": round(float(delta[1]), 4),
        "delta_angle_p05": round(float(ci[0, 0]), 4),
        "delta_angle_p95": round(float(ci[1, 0]), 4),
        "delta_distance_p05": round(float(ci[0, 1]), 4),
        "delta_distance_p95": round(float(ci[1, 1]), 4),
        "angle_improvement_fraction": round(float(np.mean(bs[:, 0] > 0)), 4),
        "distance_improvement_fraction": round(float(np.mean(bs[:, 1] < 0)), 4),
    }


def kabsch_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proper-rotation Kabsch transform (row-vector convention: ``x @ R + t ~ y``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Kabsch fit requires >= 3 finite matched atoms")
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    if np.linalg.matrix_rank(x - xc) < 2 or np.linalg.matrix_rank(y - yc) < 2:
        raise ValueError("Collinear atom selection cannot define a 3D rotation")
    u, _s, vt = np.linalg.svd((x - xc).T @ (y - yc))
    sign = np.eye(3)
    sign[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ sign @ vt
    translation = yc - xc @ rotation
    return rotation, translation


def rmsd(x: np.ndarray, y: np.ndarray) -> float:
    """Direct coordinate RMSD between matched arrays ``x`` and ``y``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=1))))


def symmetry_rmsd(
    x: np.ndarray,
    y: np.ndarray,
    descriptors: Sequence[tuple[int, str, str]],
    transform: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Aligned RMSD resolving chemically equivalent ASP OD1/OD2 permutations.

    ``descriptors`` is a sequence of ``(position, resname, atom_name)`` tuples.
    When ``transform`` is provided (e.g., global backbone Kabsch fit), applies
    that transform without re-fitting the site.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pairs: list[tuple[int, int]] = []
    for pos in sorted({d[0] for d in descriptors}):
        names = {
            name: i
            for i, (p, residue, name) in enumerate(descriptors)
            if p == pos and residue == "ASP"
        }
        if {"OD1", "OD2"} <= names.keys():
            pairs.append((names["OD1"], names["OD2"]))

    best: float | None = None
    best_flags: list[bool] = []
    for flags in itertools.product((False, True), repeat=len(pairs)):
        target = y.copy()
        for swap, (i, j) in zip(flags, pairs):
            if swap:
                target[[i, j]] = target[[j, i]]
        rot, trans = kabsch_fit(x, target) if transform is None else transform
        val = rmsd(x @ rot + trans, target)
        if best is None or val < best:
            best = val
            best_flags = list(flags)

    return {
        "rmsd_A": round(float(best if best is not None else float("nan")), 4),
        "atom_count": len(x),
        "asp_swaps": best_flags,
        "symmetry_permutations": 2 ** len(pairs),
    }


def clash_fraction(
    structure: Structure,
    protein_chain: str = "A",
    cutoff_A: float = 1.5,
) -> float:
    """Fraction of all heavy atoms participating in a protein-ligand clash (< ``cutoff_A``)."""
    if not structure.atoms:
        return 0.0
    p_coords = np.array(
        [a.xyz for a in structure.atoms if a.record_type == "ATOM" and a.chain == protein_chain],
        dtype=float,
    )
    l_coords = np.array(
        [a.xyz for a in structure.atoms if a.record_type == "HETATM"],
        dtype=float,
    )
    if p_coords.size == 0 or l_coords.size == 0:
        return 0.0
    d = np.linalg.norm(p_coords[:, None, :] - l_coords[None, :, :], axis=-1)
    ip, il = np.where(d < cutoff_A)
    n_clashing = len(np.unique(ip)) + len(np.unique(il))
    return float(n_clashing / len(structure.atoms))


def evaluate_thresholds(
    metrics: dict[str, Any],
    thresholds: ThresholdConfig,
) -> tuple[bool, list[str]]:
    """Check a flat metrics dictionary against ``ThresholdConfig``."""
    reasons: list[str] = []

    dist = metrics.get("harmonic_distance_A", metrics.get("distance_A"))
    if thresholds.max_distance_A is not None and dist is not None and dist > thresholds.max_distance_A:
        reasons.append(f"distance {dist:.3f} A > {thresholds.max_distance_A:.3f} A")

    angle = metrics.get("weighted_angle_deg", metrics.get("angle_deg"))
    if thresholds.min_angle_deg is not None and angle is not None and angle < thresholds.min_angle_deg:
        reasons.append(f"angle {angle:.1f} deg < {thresholds.min_angle_deg:.1f} deg")
    if thresholds.max_angle_deg is not None and angle is not None and angle > thresholds.max_angle_deg:
        reasons.append(f"angle {angle:.1f} deg > {thresholds.max_angle_deg:.1f} deg")

    conf = metrics.get("confidence_score_mean", metrics.get("confidence_score"))
    if thresholds.min_confidence is not None and conf is not None and conf < thresholds.min_confidence:
        reasons.append(f"confidence {conf:.3f} < {thresholds.min_confidence:.3f}")

    iptm = metrics.get("iptm_mean", metrics.get("iptm"))
    if thresholds.min_iptm is not None and iptm is not None and iptm < thresholds.min_iptm:
        reasons.append(f"iptm {iptm:.3f} < {thresholds.min_iptm:.3f}")

    lig_iptm = metrics.get("ligand_iptm_mean", metrics.get("ligand_iptm"))
    if thresholds.min_ligand_iptm is not None and lig_iptm is not None and lig_iptm < thresholds.min_ligand_iptm:
        reasons.append(f"ligand_iptm {lig_iptm:.3f} < {thresholds.min_ligand_iptm:.3f}")

    bb_rmsd = metrics.get("backbone_rmsd_A")
    if thresholds.max_backbone_rmsd_A is not None and bb_rmsd is not None and bb_rmsd > thresholds.max_backbone_rmsd_A:
        reasons.append(f"backbone_rmsd {bb_rmsd:.3f} A > {thresholds.max_backbone_rmsd_A:.3f} A")

    motif_rmsd = metrics.get("motif_rmsd_A")
    if thresholds.max_motif_rmsd_A is not None and motif_rmsd is not None and motif_rmsd > thresholds.max_motif_rmsd_A:
        reasons.append(f"motif_rmsd {motif_rmsd:.3f} A > {thresholds.max_motif_rmsd_A:.3f} A")

    clash = metrics.get("clash_fraction")
    if thresholds.max_clash_fraction is not None and clash is not None and clash > thresholds.max_clash_fraction:
        reasons.append(f"clash_fraction {clash:.4f} > {thresholds.max_clash_fraction:.4f}")

    return len(reasons) == 0, reasons
