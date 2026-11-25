"""Active-site substrate-contact detection and Boltz-2 ensemble frequency aggregation.

Extracted from ``structural_modeling_agent/analysis/substrate_contacts.py`` and
decoupled from MDAnalysis so static PDB and mmCIF cofold ensembles run with
only NumPy.

Detects functional groups on any organic substrate by covalent geometry
(carboxylate, phosphate, amine, amide_N, hydroxyl, thiol), splits non-polymer
entities into cofactor, metals, and substrate, and classifies protein contacts:
- ``salt_bridge`` (complementary charge only: Lys/Arg/HIP <-> anion; Asp/Glu <-> cation; <= 4.0 Å)
- ``h_bond`` (polar side-chain donor <-> target; <= 3.6 Å)
- ``backbone_hbond`` (backbone amide N <-> target; <= 3.6 Å)
- ``hydrophobic`` (side-chain carbon packing; <= 4.5 Å)
- ``metal_coordinated`` (functional group <-> catalytic metal; <= 2.7 Å)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from attack_geometry.structure import AtomRecord, Structure, load_structure

CATION_ATOMS: dict[str, tuple[str, ...]] = {
    "ARG": ("NH1", "NH2", "NE"),
    "LYS": ("NZ",),
    "HIP": ("ND1", "NE2"),
    "HSP": ("ND1", "NE2"),
}

HBOND_DONOR_RES: dict[str, tuple[str, ...]] = {
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
    "ASN": ("ND2",),
    "GLN": ("NE2",),
    "HIS": ("ND1", "NE2"),
    "HIE": ("ND1", "NE2"),
    "HID": ("ND1", "NE2"),
    "HSE": ("ND1", "NE2"),
    "HSD": ("ND1", "NE2"),
    "TRP": ("NE1",),
    "CYS": ("SG",),
}

HBOND_ACCEPTOR_RES: dict[str, tuple[str, ...]] = {
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
    "ASN": ("OD1",),
    "GLN": ("OE1",),
    "HIS": ("ND1", "NE2"),
    "HIE": ("ND1", "NE2"),
    "HID": ("ND1", "NE2"),
    "HSE": ("ND1", "NE2"),
    "HSD": ("ND1", "NE2"),
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
}

ACIDIC_ATOMS: dict[str, tuple[str, ...]] = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
}

HYDROPHOBIC_ATOMS: tuple[str, ...] = (
    "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE", "CZ",
)

METAL_ELEMENTS = frozenset({
    "MG", "MN", "ZN", "CA", "FE", "NI", "CO", "CU",
    "NA", "K", "CD", "MO", "W", "V", "SE",
})

COMMON_COFACTOR_RESNAMES = frozenset({
    "ATP", "ADP", "AMP", "ANP", "ACP", "GTP", "GDP", "GMP", "GNP",
    "NAD", "NAH", "NAI", "NAP", "NDP", "FAD", "FMN",
    "HEM", "HEC", "HEA", "HEB", "SAM", "SAH",
    "COA", "ACO", "PLP", "PMP", "TPP", "THF", "BTN", "B12",
    "SF4", "FES", "F3S",
})

SALT_CUT_A = 4.0
HBOND_CUT_A = 3.6
HPHOB_CUT_A = 4.5
MG_COORD_CUT_A = 2.7


def _bonded(atom: AtomRecord, candidates: Sequence[AtomRecord], cutoff: float = 1.9) -> list[AtomRecord]:
    if not candidates:
        return []
    pts = np.asarray([c.xyz for c in candidates], dtype=float)
    dists = np.linalg.norm(pts - atom.xyz, axis=1)
    return [c for c, d in zip(candidates, dists) if 0.1 < d < cutoff]


def split_ligands(
    structure: Structure,
    *,
    substrate_chains: Sequence[str] | None = None,
    cofactor_chains: Sequence[str] | None = None,
    substrate_resnames: Sequence[str] | None = None,
    cofactor_resnames: Sequence[str] | None = None,
) -> tuple[list[AtomRecord], list[AtomRecord], list[AtomRecord]]:
    """Split non-polymer HETATM records into ``(cofactor, metals, substrate)``."""
    sub_ch = set(substrate_chains or ())
    cof_ch = set(cofactor_chains or ())
    sub_rn = {r.upper() for r in (substrate_resnames or ())}
    cof_rn = {r.upper() for r in (cofactor_resnames or ())}

    groups: dict[tuple[str, str, int], list[AtomRecord]] = {}
    for a in structure.atoms:
        if a.record_type != "HETATM":
            continue
        groups.setdefault((a.chain, a.resname.upper(), a.resnum), []).append(a)

    if not groups:
        raise ValueError(f"{structure.path}: no non-polymer ligand atoms found")

    cofactor: list[AtomRecord] = []
    metals: list[AtomRecord] = []
    substrate: list[AtomRecord] = []

    organic_groups: list[tuple[tuple[str, str, int], list[AtomRecord]]] = []
    for key, atoms in sorted(groups.items()):
        chain, resname, _resnum = key
        if chain in sub_ch or resname in sub_rn:
            substrate.extend(atoms)
            continue
        if chain in cof_ch or resname in cof_rn:
            cofactor.extend(atoms)
            continue
        elems = {a.element.upper() for a in atoms}
        if elems <= METAL_ELEMENTS:
            metals.extend(atoms)
            continue
        organic_groups.append((key, atoms))

    n_organic = len(organic_groups) + (1 if substrate else 0)
    for (_chain, resname, _resnum), atoms in organic_groups:
        if resname in COMMON_COFACTOR_RESNAMES or bool({a.name for a in atoms} & {"PA", "PB", "PG"}):
            cofactor.extend(atoms)
            continue
        p_atoms = [a for a in atoms if a.element.upper() == "P"]
        o_atoms = [a for a in atoms if a.element.upper() == "O"]
        is_phospho = any(len(_bonded(p, o_atoms)) >= 3 for p in p_atoms) and len(atoms) > 6
        # Only auto-assign an unannotated phosphorylated molecule as a cofactor when
        # another organic substrate is also present; otherwise treat it as a phosphorylated substrate.
        if is_phospho and n_organic > 1:
            cofactor.extend(atoms)
        else:
            substrate.extend(atoms)

    return cofactor, metals, substrate


def detect_functional_groups(substrate_atoms: Sequence[AtomRecord]) -> list[dict[str, Any]]:
    """Detect substrate functional groups and formal charge sign from covalent geometry.

    Distinguishes carboxylates (terminal C(=O)O-, charge -1) from esters/lactones/acetals
    (C(=O)-O-C, charge 0), ketones/aldehydes (C=O, <= 1.30 A) from hydroxyls (C-OH),
    amides/nitro/aromatic nitrogens (charge 0) from aliphatic amines (charge +1),
    phosphates (charge -1), and thiols/thioethers (charge 0).
    """
    c_atoms = [a for a in substrate_atoms if a.element.upper() == "C"]
    o_atoms = [a for a in substrate_atoms if a.element.upper() == "O"]
    n_atoms = [a for a in substrate_atoms if a.element.upper() == "N"]
    s_atoms = [a for a in substrate_atoms if a.element.upper() == "S"]
    p_atoms = [a for a in substrate_atoms if a.element.upper() == "P"]

    groups: list[dict[str, Any]] = []
    assigned_o_keys: set[tuple[str, int, str]] = set()
    carbonyl_c_keys: set[tuple[str, int, str]] = set()

    for c in c_atoms:
        bo = _bonded(c, o_atoms)
        if len(bo) >= 2:
            # Check if any bonded oxygen bridges to another carbon (ester/lactone/carbonate/acetal)
            has_bridging_o = any(len(_bonded(o, c_atoms)) >= 2 for o in bo)
            if has_bridging_o:
                groups.append({"kind": "ester", "charge": 0, "atoms": [c, *bo], "center": c})
            else:
                groups.append({"kind": "carboxylate", "charge": -1, "atoms": [c, *bo], "center": c})
            carbonyl_c_keys.add((c.chain, c.resnum, c.name))
            for o in bo:
                assigned_o_keys.add((o.chain, o.resnum, o.name))
        elif len(bo) == 1:
            d_co = float(np.linalg.norm(c.xyz - bo[0].xyz))
            if d_co <= 1.30:
                carbonyl_c_keys.add((c.chain, c.resnum, c.name))
                groups.append({"kind": "carbonyl", "charge": 0, "atoms": [c, bo[0]], "center": c})
                assigned_o_keys.add((bo[0].chain, bo[0].resnum, bo[0].name))

    for p in p_atoms:
        bo = _bonded(p, o_atoms)
        if len(bo) >= 3:
            groups.append({"kind": "phosphate", "charge": -1, "atoms": [p, *bo], "center": p})
            for o in bo:
                assigned_o_keys.add((o.chain, o.resnum, o.name))

    for n in n_atoms:
        bc = _bonded(n, c_atoms)
        bo = _bonded(n, o_atoms)
        if bo:
            groups.append({"kind": "nitro", "charge": 0, "atoms": [n, *bo], "center": n})
            continue
        is_amide = any((c.chain, c.resnum, c.name) in carbonyl_c_keys for c in bc)
        is_aromatic_or_sp2 = (
            len(bc) >= 2
            and all(float(np.linalg.norm(n.xyz - c.xyz)) <= 1.39 for c in bc)
        )
        if is_amide:
            kind, charge = "amide_N", 0
        elif is_aromatic_or_sp2:
            kind, charge = "aromatic_N", 0
        else:
            kind, charge = "amine", 1
        groups.append({"kind": kind, "charge": charge, "atoms": [n], "center": n})

    for o in o_atoms:
        if (o.chain, o.resnum, o.name) in assigned_o_keys:
            continue
        bc = _bonded(o, c_atoms)
        if len(bc) == 1 and not _bonded(o, p_atoms):
            groups.append({"kind": "hydroxyl", "charge": 0, "atoms": [o], "center": o})
        elif len(bc) >= 2 and not _bonded(o, p_atoms):
            groups.append({"kind": "ether", "charge": 0, "atoms": [o], "center": o})

    for s in s_atoms:
        groups.append({"kind": "thiol", "charge": 0, "atoms": [s], "center": s})

    # Fallback when a small synthetic ligand has O/N/S atoms without C backbones
    if not groups and (o_atoms or n_atoms or s_atoms):
        for o in o_atoms:
            groups.append({"kind": "carboxylate_O", "charge": -1, "atoms": [o], "center": o})

    return groups


def _min_dist(a_coords: np.ndarray, b_coords: np.ndarray) -> float:
    if a_coords.size == 0 or b_coords.size == 0:
        return float("inf")
    d = np.linalg.norm(a_coords[:, None, :] - b_coords[None, :, :], axis=-1)
    return float(d.min())


def _classify_residue(
    res_atoms: dict[str, np.ndarray],
    resname: str,
    target_pos: np.ndarray,
    target_charge: int,
) -> tuple[str | None, float]:
    def _d(names: Sequence[str]) -> float:
        pts = [res_atoms[n] for n in names if n in res_atoms]
        if not pts:
            return float("inf")
        return _min_dist(np.asarray(pts, dtype=float), target_pos)

    if target_charge < 0 and resname in CATION_ATOMS:
        d = _d(CATION_ATOMS[resname])
        if d <= SALT_CUT_A:
            return "salt_bridge", d
    if target_charge > 0 and resname in ACIDIC_ATOMS:
        d = _d(ACIDIC_ATOMS[resname])
        if d <= SALT_CUT_A:
            return "salt_bridge", d
    if target_charge < 0 and resname in ACIDIC_ATOMS:
        return None, float("inf")
    if resname in HBOND_DONOR_RES:
        d = _d(HBOND_DONOR_RES[resname])
        if d <= HBOND_CUT_A:
            return "h_bond", d
    if target_charge >= 0 and resname in HBOND_ACCEPTOR_RES:
        d = _d(HBOND_ACCEPTOR_RES[resname])
        if d <= HBOND_CUT_A:
            return "h_bond", d
    bb_atoms = ("N", "O") if target_charge >= 0 else ("N",)
    dn = _d(bb_atoms)
    if dn <= HBOND_CUT_A:
        return "backbone_hbond", dn
    dc = _d(HYDROPHOBIC_ATOMS)
    if dc <= HPHOB_CUT_A:
        return "hydrophobic", dc
    return None, float("inf")


def analyze_pose(
    structure: str | Path | Structure,
    *,
    protein_chain: str | None = None,
    substrate_chains: Sequence[str] | None = None,
    cofactor_chains: Sequence[str] | None = None,
    substrate_resnames: Sequence[str] | None = None,
    cofactor_resnames: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Detect active-site substrate functional groups and their protein/metal contacts."""
    st = load_structure(structure) if not isinstance(structure, Structure) else structure
    cofactor, metals, substrate = split_ligands(
        st,
        substrate_chains=substrate_chains,
        cofactor_chains=cofactor_chains,
        substrate_resnames=substrate_resnames,
        cofactor_resnames=cofactor_resnames,
    )
    if not substrate:
        raise ValueError(f"{st.path}: no substrate atoms resolved")

    fgs = detect_functional_groups(substrate)
    p_atoms = [a for a in cofactor if a.element.upper() == "P"]
    if p_atoms:
        rc_pts = np.asarray([a.xyz for a in p_atoms], dtype=float)
        rc_desc = "cofactor_phosphorus"
    elif cofactor:
        rc_pts = np.asarray([a.xyz for a in cofactor], dtype=float)
        rc_desc = "cofactor_atoms"
    elif metals:
        rc_pts = np.asarray([a.xyz for a in metals], dtype=float)
        rc_desc = "metal"
    else:
        rc_pts = None
        rc_desc = None

    metal_pts = np.asarray([a.xyz for a in metals], dtype=float) if metals else np.empty((0, 3))

    for fg in fgs:
        pts = np.asarray([a.xyz for a in fg["atoms"]], dtype=float)
        fg["d_to_center"] = _min_dist(pts, rc_pts) if rc_pts is not None else None

    ranked = [g for g in fgs if g["d_to_center"] is not None]
    reactive = min(ranked, key=lambda g: g["d_to_center"]) if ranked else (fgs[0] if fgs else None)

    # Index protein residues once
    prot_residues: list[tuple[str, str, int, dict[str, np.ndarray], np.ndarray]] = []
    for r in st.residues:
        if protein_chain is not None and r.chain != protein_chain:
            continue
        amap = {name: xyz for name, xyz in zip(r.atom_names, r.xyz)}
        prot_residues.append((r.chain, r.resname.upper(), r.resnum, amap, r.xyz))

    results: list[dict[str, Any]] = []
    for fg in fgs:
        tpos = np.asarray([a.xyz for a in fg["atoms"]], dtype=float)
        contacts: list[dict[str, Any]] = []
        for chain, resname, resnum, amap, rxyz in prot_residues:
            if _min_dist(rxyz, tpos) > HPHOB_CUT_A + 0.5:
                continue
            cat, dist = _classify_residue(amap, resname, tpos, fg["charge"])
            if cat is not None:
                contacts.append(
                    {
                        "chain": chain,
                        "res": f"{resname}{resnum}",
                        "resnum": resnum,
                        "resname": resname,
                        "category": cat,
                        "dist_A": round(dist, 2),
                    }
                )
        contacts.sort(key=lambda c: c["dist_A"])
        metal_coord = bool(_min_dist(tpos, metal_pts) <= MG_COORD_CUT_A) if metal_pts.size else False
        results.append(
            {
                "kind": fg["kind"],
                "role": "reactive" if fg is reactive else "secondary",
                "charge": fg["charge"],
                "d_to_center_A": round(fg["d_to_center"], 2) if fg["d_to_center"] is not None else None,
                "metal_coordinated": metal_coord,
                "contacts": contacts,
            }
        )

    return {
        "structure": st.path,
        "reaction_center": rc_desc,
        "d_reactive_to_center_A": round(reactive["d_to_center"], 2)
        if reactive and reactive["d_to_center"] is not None
        else None,
        "n_functional_groups": len(results),
        "metals_present": len(metals),
        "groups": results,
    }


def analyze_ensemble(
    pose_paths: Sequence[str | Path],
    *,
    protein_chain: str | None = None,
    substrate_chains: Sequence[str] | None = None,
    cofactor_chains: Sequence[str] | None = None,
    substrate_resnames: Sequence[str] | None = None,
    cofactor_resnames: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate active-site contact frequencies across a Boltz-2 pose ensemble."""
    frames: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in pose_paths:
        try:
            frames.append(
                analyze_pose(
                    path,
                    protein_chain=protein_chain,
                    substrate_chains=substrate_chains,
                    cofactor_chains=cofactor_chains,
                    substrate_resnames=substrate_resnames,
                    cofactor_resnames=cofactor_resnames,
                )
            )
        except (ValueError, OSError) as exc:
            errors.append({"path": str(path), "error": str(exc)})

    n = len(frames)
    if n == 0:
        return {
            "source": "boltz_pose_ensemble",
            "n_poses_input": len(pose_paths),
            "n_structures": 0,
            "n_failed": len(errors),
            "errors": errors,
            "groups": {},
        }

    gc: dict[str, Counter] = defaultdict(Counter)
    gmc: Counter = Counter()
    gd: dict[str, list[float]] = defaultdict(list)

    for fr in frames:
        for grp in fr["groups"]:
            gk = f"{grp['role']}:{grp['kind']}"
            for key in {(c["chain"], c["res"], c["category"]) for c in grp["contacts"]}:
                gc[gk][key] += 1
            if grp["metal_coordinated"]:
                gmc[gk] += 1
            if grp["d_to_center_A"] is not None:
                gd[gk].append(float(grp["d_to_center_A"]))

    out_groups: dict[str, Any] = {}
    for gk in sorted(set(gc) | set(gmc) | set(gd)):
        out_groups[gk] = {
            "contacts": [
                {"chain": ch, "res": res, "category": cat, "frequency": round(count / n, 3), "count": count}
                for (ch, res, cat), count in sorted(gc[gk].items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "metal_coordination_freq": round(gmc[gk] / n, 3),
            "d_to_center_A_mean": round(float(np.mean(gd[gk])), 2) if gd[gk] else None,
            "d_to_center_A_std": round(float(np.std(gd[gk])), 2) if gd[gk] else None,
        }

    return {
        "source": "boltz_pose_ensemble",
        "n_poses_input": len(pose_paths),
        "n_structures": n,
        "n_failed": len(errors),
        "errors": errors,
        "groups": out_groups,
    }
