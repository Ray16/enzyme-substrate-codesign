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
    """Per-pose catalytic distance, optional 3-atom angle, and optional 4-atom signed dihedral."""

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
    dihedral_deg: float | None = None

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
        if self.dihedral_deg is not None:
            out["dihedral_deg"] = round(self.dihedral_deg, 3)
        return out


@dataclass(frozen=True)
class EnsembleGeometry:
    """Ensemble summary of per-pose distances, optional angles, and optional dihedrals."""

    n_poses: int
    mean_distance_A: float
    median_distance_A: float
    harmonic_distance_A: float
    mean_angle_deg: float | None = None
    median_angle_deg: float | None = None
    weighted_angle_deg: float | None = None
    mean_dihedral_deg: float | None = None

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
        if self.mean_dihedral_deg is not None:
            out["mean_dihedral_deg"] = round(self.mean_dihedral_deg, 3)
        return out


def _signed_dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute signed IUPAC dihedral angle (degrees in [-180, 180]) for p0-p1-p2-p3."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = float(np.linalg.norm(b1))
    if b1_norm <= 0.0:
        raise ValueError("Degenerate dihedral geometry (zero central bond vector)")
    b1_u = b1 / b1_norm
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    if float(np.linalg.norm(v)) <= 1e-12 or float(np.linalg.norm(w)) <= 1e-12:
        raise ValueError("Degenerate collinear dihedral geometry")
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_u, v), w))
    return math.degrees(math.atan2(y, x))


def evaluate_attack_geometry(
    structure: str | Structure,
    config: GeometryConfig,
) -> AttackGeometry:
    """Compute catalytic distance, optional 3-atom angle, and optional 4-atom dihedral.

    Supports both single and multi-candidate ``donor`` and ``electrophile``
    selections (choosing the closest ``(donor, electrophile)`` pair so symmetric
    substrates such as diesters, epoxides, or disulfides are handled cleanly).
    """
    st = load_structure(structure) if not isinstance(structure, Structure) else structure
    donors = select_atoms(st, config.donor)
    centers = select_atoms(st, config.electrophile)

    donor_coords = np.asarray([a.xyz for a in donors], dtype=float)
    center_coords = np.asarray([a.xyz for a in centers], dtype=float)
    if not (np.isfinite(donor_coords).all() and np.isfinite(center_coords).all()):
        raise ValueError(f"{st.path}: non-finite coordinates in selected atoms")

    dist_matrix = np.linalg.norm(donor_coords[:, None, :] - center_coords[None, :, :], axis=-1)
    flat_best = int(np.argmin(dist_matrix))
    best_d_idx, best_c_idx = divmod(flat_best, len(centers))
    best_donor = donors[best_d_idx]
    center = centers[best_c_idx]
    d = float(dist_matrix[best_d_idx, best_c_idx])

    angle: float | None = None
    leaving_chain: str | None = None
    leaving_atom: str | None = None
    leaving = None
    if config.leaving is not None:
        leavings = select_atoms(st, config.leaving)
        if len(leavings) == 1:
            leaving = leavings[0]
        else:
            # For symmetric substrates with multiple equivalent leaving/reference atoms,
            # pick the leaving atom nearest to the selected electrophile center.
            l_coords = np.asarray([a.xyz for a in leavings], dtype=float)
            if not np.isfinite(l_coords).all():
                raise ValueError(f"{st.path}: non-finite coordinates in angle reference atoms")
            l_dists = np.linalg.norm(l_coords - center.xyz, axis=1)
            leaving = leavings[int(np.argmin(l_dists))]
        if not np.isfinite(leaving.xyz).all():
            raise ValueError(f"{st.path}: non-finite coordinates in angle reference atom")
        u = donor_coords[best_d_idx] - center.xyz
        v = leaving.xyz - center.xyz
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        if denom <= 0.0:
            raise ValueError(f"{st.path}: degenerate geometry (zero vector norm)")
        cos_theta = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
        angle = math.degrees(math.acos(cos_theta))
        leaving_chain = leaving.chain
        leaving_atom = leaving.name

    dihedral: float | None = None
    if config.dihedral_ref is not None:
        if leaving is None:
            raise ValueError(
                f"{st.path}: 'dihedral_ref' requires 'leaving' to define a 4-atom dihedral."
            )
        drefs = select_atoms(st, config.dihedral_ref)
        if len(drefs) == 1:
            dref = drefs[0]
        else:
            dr_coords = np.asarray([a.xyz for a in drefs], dtype=float)
            dref = drefs[int(np.argmin(np.linalg.norm(dr_coords - center.xyz, axis=1)))]
        if not np.isfinite(dref.xyz).all():
            raise ValueError(f"{st.path}: non-finite coordinates in dihedral reference atom")
        dihedral = _signed_dihedral_deg(donor_coords[best_d_idx], center.xyz, leaving.xyz, dref.xyz)

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
        n_candidates=len(donors) * len(centers),
        dihedral_deg=dihedral,
    )


def evaluate_extra_metrics(
    structure: str | Structure,
    config: GeometryConfig,
) -> dict[str, float]:
    """Evaluate all optional auxiliary distances/angles/dihedrals in ``config.extra_metrics``."""
    if not config.extra_metrics:
        return {}
    st = load_structure(structure) if not isinstance(structure, Structure) else structure
    out: dict[str, float] = {}
    for spec in config.extra_metrics:
        sub_cfg = GeometryConfig(
            donor=spec.donor,
            electrophile=spec.electrophile,
            leaving=spec.leaving,
            dihedral_ref=spec.dihedral_ref,
            clash_cutoff_A=config.clash_cutoff_A,
        )
        g = evaluate_attack_geometry(st, sub_cfg)
        out[f"{spec.name}_distance_A"] = round(g.distance_A, 4)
        if g.angle_deg is not None:
            out[f"{spec.name}_angle_deg"] = round(g.angle_deg, 3)
        if g.dihedral_deg is not None:
            out[f"{spec.name}_dihedral_deg"] = round(g.dihedral_deg, 3)
    return out


def ranking_sort_key(
    row: dict[str, Any],
    *,
    id_key: str,
    rank_by: str = "distance",
    target_angle_deg: float | None = None,
) -> tuple:
    """Mechanism-aware sort key for ranking evaluated variants or designs.

    Supports:
    - ``angle_error``: minimizes ``|angle - target_angle_deg|`` (e.g. 180° for inline SN2,
      107° for Bürgi-Dunitz carbonyl addition), breaking ties by shorter distance & higher confidence.
    - ``distance``: minimizes ``harmonic_distance_A`` / ``mean_distance_A``.
    - ``confidence``: maximizes ``confidence_score_mean``.
    - ``iptm``: maximizes ``iptm_mean`` / ``ligand_iptm_mean``.
    """
    evaluated = row.get("status") == "evaluated"
    passed = bool(row.get("passed_thresholds", False))
    dist = float(row.get("harmonic_distance_A", row.get("mean_distance_A", 1e9))) if evaluated else 1e9
    angle = row.get("weighted_angle_deg", row.get("mean_angle_deg"))
    conf = float(row.get("confidence_score_mean", -1.0)) if evaluated else -1.0
    iptm = float(row.get("iptm_mean", row.get("ligand_iptm_mean", -1.0))) if evaluated else -1.0

    if angle is not None and target_angle_deg is not None:
        angle_err = abs(float(angle) - float(target_angle_deg))
    elif angle is not None:
        angle_err = -float(angle)
    else:
        angle_err = 1e9

    if rank_by == "angle_error":
        primary = (angle_err, dist, -conf)
    elif rank_by == "confidence":
        primary = (-conf, dist, angle_err)
    elif rank_by == "iptm":
        primary = (-iptm, dist, -conf)
    else:  # "distance"
        primary = (dist, angle_err, -conf)

    return (not passed, not evaluated, *primary, str(row.get(id_key, "")))


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


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    rads = np.deg2rad(angles_deg)
    return float(math.degrees(math.atan2(float(np.mean(np.sin(rads))), float(np.mean(np.cos(rads))))))


def summarize_ensemble(geometries: Sequence[AttackGeometry]) -> EnsembleGeometry:
    """Aggregate per-pose geometries into standard and inverse-distance ensemble statistics."""
    if not geometries:
        raise ValueError("Cannot summarize an empty pose ensemble")
    dists = np.array([g.distance_A for g in geometries], dtype=float)
    h_dist = float(len(dists) / np.sum(1.0 / dists))

    mean_dihed: float | None = None
    if all(g.dihedral_deg is not None for g in geometries):
        mean_dihed = _circular_mean_deg(np.array([float(g.dihedral_deg) for g in geometries], dtype=float))

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
            mean_dihedral_deg=mean_dihed,
        )

    return EnsembleGeometry(
        n_poses=len(geometries),
        mean_distance_A=float(np.mean(dists)),
        median_distance_A=float(np.median(dists)),
        harmonic_distance_A=h_dist,
        mean_dihedral_deg=mean_dihed,
    )


def bootstrap_wt_delta(
    variant_poses: np.ndarray,
    wt_poses: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
    target_angle_deg: float | None = None,
) -> dict[str, float]:
    """Independent ensemble bootstrap comparison against matched wild-type poses.

    Supports both:
    - 2D ``(n_poses, 2)`` arrays of ``(distance_A, angle_deg)``, and
    - 1D ``(n_poses,)`` (or ``(n_poses, 1)``) arrays of ``distance_A`` for
      distance-only mechanisms (e.g., ``general_acid_base``, ``metalloenzyme_coordination``).

    When ``target_angle_deg`` is provided (e.g., ``107.0`` for Bürgi-Dunitz
    carbonyl addition or ``180.0`` for inline SN2), ``angle_improvement_fraction``
    measures the bootstrap probability that ``|angle - target_angle_deg|``
    decreased relative to WT, and ``delta_angle_error_deg`` reports the signed
    change in error (negative = closer to target).
    """
    var_arr = np.asarray(variant_poses, dtype=float)
    wt_arr = np.asarray(wt_poses, dtype=float)
    rng = np.random.default_rng(seed)

    # Distance-only bootstrap path
    if var_arr.ndim == 1 or (var_arr.ndim == 2 and var_arr.shape[-1] == 1):
        vd = var_arr.reshape(-1)
        wd = wt_arr.reshape(-1)
        if vd.size == 0 or wd.size == 0 or np.any(vd <= 0) or np.any(wd <= 0):
            raise ValueError("Expected non-empty positive distances for bootstrap_wt_delta")
        v_h = float(len(vd) / np.sum(1.0 / vd))
        w_h = float(len(wd) / np.sum(1.0 / wd))
        bs_v = vd[rng.integers(len(vd), size=(n_bootstrap, len(vd)))]
        bs_w = wd[rng.integers(len(wd), size=(n_bootstrap, len(wd)))]
        bs_d = (bs_v.shape[-1] / np.sum(1.0 / bs_v, axis=-1)) - (bs_w.shape[-1] / np.sum(1.0 / bs_w, axis=-1))
        ci = np.percentile(bs_d, [5, 95])
        return {
            "delta_distance_A": round(v_h - w_h, 4),
            "delta_distance_p05": round(float(ci[0]), 4),
            "delta_distance_p95": round(float(ci[1]), 4),
            "distance_improvement_fraction": round(float(np.mean(bs_d < 0)), 4),
        }

    var_summary = proximity_weighted_angle_and_distance(var_arr)
    wt_summary = proximity_weighted_angle_and_distance(wt_arr)
    delta = var_summary - wt_summary
    bs_var = proximity_weighted_angle_and_distance(
        var_arr[rng.integers(len(var_arr), size=(n_bootstrap, len(var_arr)))]
    )
    bs_wt = proximity_weighted_angle_and_distance(
        wt_arr[rng.integers(len(wt_arr), size=(n_bootstrap, len(wt_arr)))]
    )
    bs = bs_var - bs_wt
    ci = np.percentile(bs, [5, 95], axis=0)

    out: dict[str, float] = {
        "delta_angle_deg": round(float(delta[0]), 4),
        "delta_distance_A": round(float(delta[1]), 4),
        "delta_angle_p05": round(float(ci[0, 0]), 4),
        "delta_angle_p95": round(float(ci[1, 0]), 4),
        "delta_distance_p05": round(float(ci[0, 1]), 4),
        "delta_distance_p95": round(float(ci[1, 1]), 4),
        "distance_improvement_fraction": round(float(np.mean(bs[:, 1] < 0)), 4),
    }
    if target_angle_deg is not None:
        t = float(target_angle_deg)
        err_delta = abs(float(var_summary[0]) - t) - abs(float(wt_summary[0]) - t)
        bs_err_delta = np.abs(bs_var[:, 0] - t) - np.abs(bs_wt[:, 0] - t)
        out["delta_angle_error_deg"] = round(err_delta, 4)
        out["angle_improvement_fraction"] = round(float(np.mean(bs_err_delta < 0)), 4)
    else:
        out["angle_improvement_fraction"] = round(float(np.mean(bs[:, 0] > 0)), 4)
    return out


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
    protein_chain: str | Sequence[str] | None = None,
    cutoff_A: float = 1.5,
) -> float:
    """Fraction of heavy atoms participating in a protein-ligand clash (< ``cutoff_A``).

    When ``protein_chain`` is ``None``, checks all polymer ``ATOM`` chains against
    all ``HETATM`` atoms so oligomeric/partner-chain clashes are also caught.
    Pass a chain ID string or sequence of chain IDs to restrict to specific chains.
    """
    if not structure.atoms:
        return 0.0
    if protein_chain is None:
        chains: set[str] | None = None
    elif isinstance(protein_chain, str):
        chains = {protein_chain}
    else:
        chains = set(protein_chain)

    p_coords = np.array(
        [
            a.xyz
            for a in structure.atoms
            if a.record_type == "ATOM" and (chains is None or a.chain in chains)
        ],
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

    dihed = metrics.get("mean_dihedral_deg", metrics.get("dihedral_deg"))
    if thresholds.min_dihedral_deg is not None and dihed is not None and dihed < thresholds.min_dihedral_deg:
        reasons.append(f"dihedral {dihed:.1f} deg < {thresholds.min_dihedral_deg:.1f} deg")
    if thresholds.max_dihedral_deg is not None and dihed is not None and dihed > thresholds.max_dihedral_deg:
        reasons.append(f"dihedral {dihed:.1f} deg > {thresholds.max_dihedral_deg:.1f} deg")

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


def evaluate_extra_thresholds(
    metrics: dict[str, Any],
    config: GeometryConfig,
) -> list[str]:
    """Check any threshold constraints declared on ``config.extra_metrics``."""
    reasons: list[str] = []
    for spec in config.extra_metrics:
        d = metrics.get(f"{spec.name}_distance_A")
        if spec.max_distance_A is not None and d is not None and d > spec.max_distance_A:
            reasons.append(f"{spec.name}_distance {d:.3f} A > {spec.max_distance_A:.3f} A")
        a = metrics.get(f"{spec.name}_angle_deg")
        if spec.min_angle_deg is not None and a is not None and a < spec.min_angle_deg:
            reasons.append(f"{spec.name}_angle {a:.1f} deg < {spec.min_angle_deg:.1f} deg")
        if spec.max_angle_deg is not None and a is not None and a > spec.max_angle_deg:
            reasons.append(f"{spec.name}_angle {a:.1f} deg > {spec.max_angle_deg:.1f} deg")
        dh = metrics.get(f"{spec.name}_dihedral_deg")
        if spec.min_dihedral_deg is not None and dh is not None and dh < spec.min_dihedral_deg:
            reasons.append(f"{spec.name}_dihedral {dh:.1f} deg < {spec.min_dihedral_deg:.1f} deg")
        if spec.max_dihedral_deg is not None and dh is not None and dh > spec.max_dihedral_deg:
            reasons.append(f"{spec.name}_dihedral {dh:.1f} deg > {spec.max_dihedral_deg:.1f} deg")
    return reasons
