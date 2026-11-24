"""Synthetic holo PDB/mmCIF fixtures and mock Boltz-2 outputs for offline tests."""
from __future__ import annotations

import json
from pathlib import Path

RESIDUES = ["ALA", "ASP", "GLY", "VAL", "THR"]
SEQUENCE = "ADGVT"


def _pdb_atom(serial: int, name: str, resname: str, chain: str, resnum: int, xyz: tuple[float, float, float], elem: str, het: bool = False) -> str:
    rec = "HETATM" if het else "ATOM  "
    x, y, z = xyz
    return (
        f"{rec}{serial:5d} {name:^4s}{resname:>4s} {chain}{resnum:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 85.00          {elem:>2s}"
    )


def write_synthetic_holo_pdb(
    path: str | Path,
    *,
    seq: str = SEQUENCE,
    donor_offset: tuple[float, float, float] = (-3.0, 0.0, 0.0),
    swap_asp_od: bool = False,
    stray_metal: bool = True,
    bridging_metal: bool = True,
) -> Path:
    """Write a synthetic holo enzyme PDB with known reactive geometry.

    Electrophile PA is at (16.5, 4.0, 0.0), leaving O3A is at (18.0, 4.0, 0.0)
    (+x direction from PA). Default donor O1 is at PA + (-3.0, 0.0, 0.0) = (13.5, 4.0, 0.0),
    giving exact distance = 3.0 Å and exact inline attack angle = 180.0 deg.
    """
    one_to_three = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
        "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
        "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    }
    lines: list[str] = []
    serial = 1
    for resnum, aa in enumerate(seq, start=1):
        resname = one_to_three[aa]
        x0 = resnum * 8.0
        for name, elem, off in (
            ("N", "N", (0.0, 0.0, 0.0)),
            ("CA", "C", (1.2, 0.5, 0.0)),
            ("C", "C", (2.4, 0.0, 0.5)),
            ("O", "O", (2.4, -1.1, 0.5)),
        ):
            lines.append(_pdb_atom(serial, name, resname, "A", resnum, (x0 + off[0], off[1], off[2]), elem))
            serial += 1
        if resname == "ASP":
            od1 = (x0 + 1.2, 2.5, 0.8)
            od2 = (x0 + 1.2, 2.5, -0.8)
            if swap_asp_od:
                od1, od2 = od2, od1
            lines.append(_pdb_atom(serial, "CG", resname, "A", resnum, (x0 + 1.2, 1.8, 0.0), "C"))
            serial += 1
            lines.append(_pdb_atom(serial, "OD1", resname, "A", resnum, od1, "O"))
            serial += 1
            lines.append(_pdb_atom(serial, "OD2", resname, "A", resnum, od2, "O"))
            serial += 1

    # Cofactor (chain B): electrophile PA and leaving atom O3A
    pa = (16.5, 4.0, 0.0)
    o3a = (18.0, 4.0, 0.0)
    lines.append(_pdb_atom(serial, "PA", "ATP", "B", 1, pa, "P", het=True))
    serial += 1
    lines.append(_pdb_atom(serial, "O3A", "ATP", "B", 1, o3a, "O", het=True))
    serial += 1

    # Substrate (chain D): 2 nucleophilic oxygens (O1 is closer)
    o1 = (pa[0] + donor_offset[0], pa[1] + donor_offset[1], pa[2] + donor_offset[2])
    o2 = (o1[0] - 2.0, o1[1] + 1.5, o1[2])
    lines.append(_pdb_atom(serial, "O1", "SUB", "D", 1, o1, "O", het=True))
    serial += 1
    lines.append(_pdb_atom(serial, "O2", "SUB", "D", 1, o2, "O", het=True))
    serial += 1

    if bridging_metal:
        # Within 5 A of chain B (O3A at x=18) and within 5 A of residue 3 (x=24)
        lines.append(_pdb_atom(serial, "MG", "MG", "C", 1, (21.5, 2.0, 0.0), "MG", het=True))
        serial += 1
    if stray_metal:
        lines.append(_pdb_atom(serial, "MG", "MG", "Z", 1, (100.0, 0.0, 0.0), "MG", het=True))
        serial += 1

    # Water (should be ignored by default)
    lines.append(_pdb_atom(serial, "O", "HOH", "W", 1, (16.5, 6.0, 0.0), "O", het=True))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\nEND\n")
    return p


def write_synthetic_holo_cif(path: str | Path, seq: str = SEQUENCE) -> Path:
    """Write a minimal mmCIF matching ``write_synthetic_holo_pdb``."""
    pdb_tmp = Path(path).with_suffix(".tmp.pdb")
    write_synthetic_holo_pdb(pdb_tmp, seq=seq)
    rows = []
    atom_id = 1
    for line in pdb_tmp.read_text().splitlines():
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        group = line[:6].strip()
        name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip()
        resnum = int(line[22:26])
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        elem = line[76:78].strip()
        rows.append(f"{group} {atom_id} {elem} {name} {resname} {chain} {resnum} {x:.3f} {y:.3f} {z:.3f} 85.0 1")
        atom_id += 1
    pdb_tmp.unlink()

    cif_lines = [
        "data_synthetic",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.auth_atom_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
        *rows,
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(cif_lines) + "\n")
    return p


def write_mock_boltz_prediction(
    results_root: Path,
    design_id: str,
    seq: str,
    *,
    offsets: list[tuple[float, float, float]],
    confidence_score: float = 0.85,
    swap_asp_od: bool = False,
) -> Path:
    """Create a stock Boltz-2 prediction directory with PDB models + confidence JSONs."""
    pred_dir = results_root / design_id / f"boltz_results_{design_id}" / "predictions" / design_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    for idx, off in enumerate(offsets):
        write_synthetic_holo_pdb(
            pred_dir / f"{design_id}_model_{idx}.pdb",
            seq=seq,
            donor_offset=off,
            swap_asp_od=swap_asp_od,
            stray_metal=False,
        )
        conf = {
            "confidence_score": confidence_score,
            "iptm": confidence_score - 0.05,
            "ligand_iptm": confidence_score - 0.08,
            "complex_plddt": 0.90,
            "complex_iplddt": 0.88,
            "pair_chains_iptm": {
                "0": {"0": 0.95, "2": confidence_score - 0.02},
                "2": {"0": confidence_score - 0.10, "2": 0.92},
            },
        }
        (pred_dir / f"confidence_{design_id}_model_{idx}.json").write_text(json.dumps(conf, indent=2))
    return pred_dir
