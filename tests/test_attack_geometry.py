"""Tests for shared attack-geometry evaluation, structure parsing, and Kabsch/ASP symmetry RMSD."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from attack_geometry.cli import main as geom_cli_main
from attack_geometry.config import GeometryConfig, ThresholdConfig
from attack_geometry.metrics import (
    clash_fraction,
    evaluate_attack_geometry,
    evaluate_thresholds,
    proximity_weighted_angle_and_distance,
    summarize_ensemble,
    symmetry_rmsd,
)
from attack_geometry.structure import load_structure

from ._synthetic import SEQUENCE, write_synthetic_holo_cif, write_synthetic_holo_pdb


def _default_geom_cfg() -> GeometryConfig:
    return GeometryConfig.from_dict(
        {
            "donor": {"chain": "D", "elements": ["O"], "n_atoms": 2},
            "electrophile": {"chain": "B", "atom": "PA"},
            "leaving": {"chain": "B", "atom": "O3A"},
        }
    )


def test_pdb_and_cif_attack_geometry_exact(tmp_path: Path):
    pdb = write_synthetic_holo_pdb(tmp_path / "holo.pdb", donor_offset=(-3.0, 0.0, 0.0))
    cif = write_synthetic_holo_cif(tmp_path / "holo.cif")
    cfg = _default_geom_cfg()

    g_pdb = evaluate_attack_geometry(pdb, cfg)
    g_cif = evaluate_attack_geometry(cif, cfg)

    assert g_pdb.distance_A == pytest.approx(3.0, abs=1e-4)
    assert g_pdb.angle_deg == pytest.approx(180.0, abs=1e-3)
    assert g_pdb.attack_atom == "O1"
    assert g_cif.distance_A == pytest.approx(g_pdb.distance_A, abs=1e-4)
    assert g_cif.angle_deg == pytest.approx(g_pdb.angle_deg, abs=1e-3)


def test_attack_angle_90_deg(tmp_path: Path):
    pdb = write_synthetic_holo_pdb(tmp_path / "perp.pdb", donor_offset=(0.0, 2.5, 0.0))
    g = evaluate_attack_geometry(pdb, _default_geom_cfg())
    assert g.distance_A == pytest.approx(2.5, abs=1e-4)
    assert g.angle_deg == pytest.approx(90.0, abs=1e-3)


def test_proximity_weighted_ensemble():
    poses = np.array([[2.0, 160.0], [4.0, 100.0]])
    w_angle, h_dist = proximity_weighted_angle_and_distance(poses)
    # weights 1/2 and 1/4 -> 2/3 * 160 + 1/3 * 100 = 140
    assert w_angle == pytest.approx(140.0, abs=1e-6)
    assert h_dist == pytest.approx(2.0 / (0.5 + 0.25), abs=1e-6)


def test_asp_od1_od2_symmetry_rmsd(tmp_path: Path):
    p1 = load_structure(write_synthetic_holo_pdb(tmp_path / "a.pdb", swap_asp_od=False))
    p2 = load_structure(write_synthetic_holo_pdb(tmp_path / "b.pdb", swap_asp_od=True))
    m1 = p1.atom_map("A")
    m2 = p2.atom_map("A")
    keys = [(2, "N"), (2, "C"), (2, "CG"), (2, "OD1"), (2, "OD2")]
    x = np.array([m1[k] for k in keys])
    y = np.array([m2[k] for k in keys])
    desc = [(2, "ASP", "N"), (2, "ASP", "C"), (2, "ASP", "CG"), (2, "ASP", "OD1"), (2, "ASP", "OD2")]
    res = symmetry_rmsd(x, y, desc)
    assert res["rmsd_A"] == pytest.approx(0.0, abs=1e-4)
    assert res["symmetry_permutations"] == 2
    assert res["asp_swaps"] == [True]


def test_geometry_cli(tmp_path: Path):
    pdb = write_synthetic_holo_pdb(tmp_path / "holo.pdb", donor_offset=(-3.2, 0.0, 0.0))
    cfg_path = tmp_path / "geom.json"
    cfg_path.write_text(json.dumps({"geometry": _default_geom_cfg().to_dict()}))
    out_path = tmp_path / "out.json"

    rc = geom_cli_main([str(pdb), "--config", str(cfg_path), "--output", str(out_path)])
    assert rc == 0
    data = json.loads(out_path.read_text())
    assert data["ensemble"]["harmonic_distance_A"] == pytest.approx(3.2, abs=1e-3)
    assert data["ensemble"]["passed"] is True
