"""General catalytic geometry evaluation, ensemble aggregation, and symmetry-aware Kabsch RMSD.

Supports arbitrary user-configured atom selectors for distance (and optional
3-atom angle) evaluation, standard & proximity-weighted ensemble summaries,
Kabsch superposition with general chemically equivalent sidechain atom pairs
(ASP, GLU, PHE, TYR, ARG, VAL, LEU, or custom pairs), and configurable steric
clash checks.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .config import GeometryConfig, ThresholdConfig
from .structure import Structure, load_structure, select_atoms

#: Standard chemically equivalent sidechain heavy-atom pairs for symmetry-aware RMSD.
STANDARD_SYMMETRY_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "ARG": (("NH1", "NH2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
    "VAL": (("CG1", "CG2"),),
    "LEU": (("CD1", "CD2"),),
}


@dataclass(frozen=True)
class AttackGeometry:
    """Per-pose catalytic distance and optional 3-atom angle."""

    distance_A: float
    angle_deg: float | None
    attack_chain: str
    attack_resnum: int
    attack_atom: str
    center_chain: str
    center_atom: str
    leaving_chain: str | None
    leaving_atom: str | None
    n_candidates: int

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "distance_A": round(self.distance_A, 4),
            "angle_deg": round(self.angle_deg, 3) if self.angle_deg is not None else None,
            "attack_chain": self.attack_chain,
            "attack_resnum": self.attack_resnum,
            "attack_atom": self.attack_atom,
            "center_chain": self.center_chain,
            "center_atom": self.center_atom,
            "n_candidates": self.n_candidates,
        }
        if self.leaving_atom is not None:
            out["leaving_chain"] = self.leaving_chain
            out["leaving_atom"] = self.leaving_atom
        return out


@dataclass(frozen=True)
class EnsembleGeometry:
    """Ensemble summary of per-pose distances and (optional) angles."""

    n_poses: int
    mean_distance_A: float
    median_distance_A: float
    harmonic_distance_A: float
    mean_angle_deg: float | None = None
    median_angle_deg: float | None = None
    weighted_angle_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_poses": self.n_poses,
            "mean_distance_A": round(self.mean_distance_A, 4),
            "median_distance_A": round(self.median_distance_A, 4),
            "harmonic_distance_A": round(self.harmonic_distance_A, 4),
        }
        if self.mean_angle_deg is not None:
            out["mean_angle_deg"] = round(self.mean_angle_deg, 3)
            out["median_angle_deg"] = round(self.median_angle_deg, 3)
            out["weighted_angle_deg"] = round(self.weighted_angle_deg, 3)
        return out


def evaluate_attack_geometry(
    structure: str | Structure,
    config: GeometryConfig,
) -> AttackGeometry:
    """Compute distance between nearest ``donor`` atom and ``electrophile``, plus optional angle."""
    st = load_structure(structure) if not isinstance(structure, Structure) else structure
    donors = select_atoms(st, config.donor)
    centers = select_atoms(st, config.electrophile)
    if len(centers) != 1:
        raise ValueError(
            f"{st.path}: electrophile/center selector matched {len(centers)} atoms; expected 1."
        )

    center = centers[0]
    donor_coords = np.asarray([a.xyz for a in donors], dtype=float)
    if not (np.isfinite(donor_coords).all() and np.isfinite(center.xyz).all()):
        raise ValueError(f"{st.path}: non-finite coordinates in selected atoms")

    distances = np.linalg.norm(donor_coords - center.xyz, axis=1)
    best_idx = int(np.argmin(distances))
    best_donor = donors[best_idx]
    d = float(distances[best_idx])

    angle: float | None = None
    leaving_chain: str | None = None
    leaving_atom: str | None = None
    if config.leaving is not None:
        leavings = select_atoms(st, config.leaving)
        if len(leavings) != 1:
            raise ValueError(
                f"{st.path}: leaving/angle_reference selector matched {len(leavings)} atoms; expected 1."
            )
        leaving = leavings[0]
        if not np.isfinite(leaving.xyz).all():
            raise ValueError(f"{st.path}: non-finite coordinates in angle reference atom")
        u = donor_coords[best_idx] - center.xyz
        v = leaving.xyz - center.xyz
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        if denom <= 0.0:
            raise ValueError(f"{st.path}: degenerate geometry (zero vector norm)")
        cos_theta = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
        angle = math.degrees(math.acos(cos_theta))
        leaving_chain = leaving.chain
        leaving_atom = leaving.name

    return AttackGeometry(
        distance_A=d,
        angle_deg=angle,
        attack_chain=best_donor.chain,
        attack_resnum=best_donor.resnum,
        attack_atom=best_donor.name,
        center_chain=center.chain,
        center_atom=center.name,
        leaving_chain=leaving_chain,
        leaving_atom=leaving_atom,
        n_candidates=len(donors),
    )


def proximity_weighted_angle_and_distance(poses: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return ``(..., 2)`` array of ``(weighted_angle_deg, harmonic_distance_A)``.

    Optional inverse-distance weighted summary where ``poses[..., 0]`` is
    distance (Å) and ``poses[..., 1]`` is angle (degrees).
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
    """Aggregate per-pose geometries into standard and inverse-distance ensemble statistics."""
    if not geometries:
        raise ValueError("Cannot summarize an empty pose ensemble")
    dists = np.array([g.distance_A for g in geometries], dtype=float)
    h_dist = float(len(dists) / np.sum(1.0 / dists))

    if all(g.angle_deg is not None for g in geometries):
        angles = np.array([float(g.angle_deg) for g in geometries], dtype=float)
        arr = np.column_stack([dists, angles])
        weighted = proximity_weighted_angle_and_distance(arr)
        return EnsembleGeometry(
            n_poses=len(geometries),
            mean_distance_A=float(np.mean(dists)),
            median_distance_A=float(np.median(dists)),
            harmonic_distance_A=h_dist,
            mean_angle_deg=float(np.mean(angles)),
            median_angle_deg=float(np.median(angles)),
            weighted_angle_deg=float(weighted[0]),
        )

    return EnsembleGeometry(
        n_poses=len(geometries),
        mean_distance_A=float(np.mean(dists)),
        median_distance_A=float(np.median(dists)),
        harmonic_distance_A=h_dist,
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
    equivalent_pairs: Mapping[str, Sequence[tuple[str, str]]] = STANDARD_SYMMETRY_PAIRS,
) -> dict[str, Any]:
    """Aligned RMSD resolving chemically equivalent sidechain atom permutations.

    ``descriptors`` is a sequence of ``(position, resname, atom_name)`` tuples.
    By default resolves all standard symmetric sidechain pairs (ASP, GLU, ARG,
    PHE, TYR, VAL, LEU); pass ``equivalent_pairs={}`` to disable symmetry swaps
    or supply custom residue->pair mappings.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Group coupled flips per residue (e.g., PHE/TYR flip CD1<->CD2 and CE1<->CE2 together)
    swap_groups: list[list[tuple[int, int]]] = []
    for pos in sorted({d[0] for d in descriptors}):
        resnames = {r for p, r, _ in descriptors if p == pos}
        for resname in sorted(resnames):
            atom_idx = {
                name: i
                for i, (p, r, name) in enumerate(descriptors)
                if p == pos and r == resname
            }
            pairs_for_res = [
                (atom_idx[a1], atom_idx[a2])
                for a1, a2 in equivalent_pairs.get(resname, ())
                if a1 in atom_idx and a2 in atom_idx
            ]
            if pairs_for_res:
                swap_groups.append(pairs_for_res)

    best: float | None = None
    best_flags: list[bool] = []
    for flags in itertools.product((False, True), repeat=len(swap_groups)):
        target = y.copy()
        for swap, group_pairs in zip(flags, swap_groups):
            if swap:
                for i, j in group_pairs:
                    target[[i, j]] = target[[j, i]]
        rot, trans = kabsch_fit(x, target) if transform is None else transform
        val = rmsd(x @ rot + trans, target)
        if best is None or val < best:
            best = val
            best_flags = list(flags)

    return {
        "rmsd_A": round(float(best if best is not None else float("nan")), 4),
        "atom_count": len(x),
        "symmetry_swaps": best_flags,
        "symmetry_permutations": 2 ** len(swap_groups),
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
