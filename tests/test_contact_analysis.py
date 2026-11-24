"""Offline tests for active-site contact analysis, heavy-atom v2 audit, and visualization."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from contact_analysis.cli import build_parser, main as ca_cli_main
from contact_analysis.heavy_atom_audit import (
    Atom,
    ContactPolicy,
    TargetGroup,
    collapse_for_display,
    distinct_residue_count,
    evaluate_contacts,
)
from contact_analysis.substrate_contacts import analyze_ensemble, analyze_pose
from contact_analysis.visualize import (
    plot_attack_geometry_summary,
    render_contact_figure,
    write_contact_report,
)


def _write_contact_pdb(path: Path) -> Path:
    """Write a synthetic active site with Lys salt bridge, Ser H-bond, ATP, Mg, and dicarboxylate."""
    lines = [
        # Protein chain A: LYS 10 (NZ near carboxylate O1 at 2.9 A), SER 20 (OG near O2 at 3.0 A)
        "ATOM      1  N   LYS A  10      10.000   0.000   0.000  1.00 90.00           N",
        "ATOM      2  CA  LYS A  10      11.200   0.000   0.000  1.00 90.00           C",
        "ATOM      3  C   LYS A  10      12.400   0.000   0.000  1.00 90.00           C",
        "ATOM      4  O   LYS A  10      12.400  -1.100   0.000  1.00 90.00           O",
        "ATOM      5  NZ  LYS A  10      13.500   1.100   0.000  1.00 90.00           N",
        "ATOM      6  N   SER A  20      18.000   0.000   0.000  1.00 90.00           N",
        "ATOM      7  CA  SER A  20      19.200   0.000   0.000  1.00 90.00           C",
        "ATOM      8  C   SER A  20      20.400   0.000   0.000  1.00 90.00           C",
        "ATOM      9  O   SER A  20      20.400  -1.100   0.000  1.00 90.00           O",
        "ATOM     10  OG  SER A  20      15.500   6.000   0.000  1.00 90.00           O",
        # ATP cofactor (chain B)
        "HETATM   11  PA  ATP B   1      16.500   4.000   0.000  1.00 90.00           P",
        "HETATM   12  O3A ATP B   1      18.000   4.000   0.000  1.00 90.00           O",
        # Mg cofactor (chain C) coordinating O1 at 2.1 A
        "HETATM   13  MG  MG  C   1      13.500   6.100   0.000  1.00 90.00          MG",
        # Substrate carboxylate (chain D): C1 bonded to O1 (13.5, 4.0, 0.0) and O2 (14.5, 5.2, 0.0)
        "HETATM   14  C1  SUB D   1      13.800   4.700   0.000  1.00 90.00           C",
        "HETATM   15  O1  SUB D   1      13.500   4.000   0.000  1.00 90.00           O",
        "HETATM   16  O2  SUB D   1      14.500   5.200   0.000  1.00 90.00           O",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_substrate_contacts_pose_and_ensemble(tmp_path: Path):
    p1 = _write_contact_pdb(tmp_path / "pose_0.pdb")
    p2 = _write_contact_pdb(tmp_path / "pose_1.pdb")

    single = analyze_pose(p1)
    assert single["n_functional_groups"] == 1
    assert single["metals_present"] == 1
    grp = single["groups"][0]
    assert grp["kind"] == "carboxylate"
    assert grp["role"] == "reactive"
    assert grp["metal_coordinated"] is True
    cats = {c["res"]: c["category"] for c in grp["contacts"]}
    assert cats["LYS10"] == "salt_bridge"
    assert cats["SER20"] == "h_bond"

    ens = analyze_ensemble([p1, p2])
    assert ens["n_structures"] == 2
    g_ens = ens["groups"]["reactive:carboxylate"]
    assert g_ens["metal_coordination_freq"] == pytest.approx(1.0)
    freqs = {c["res"]: c["frequency"] for c in g_ens["contacts"]}
    assert freqs["LYS10"] == pytest.approx(1.0)
    assert freqs["SER20"] == pytest.approx(1.0)


def test_heavy_atom_v2_audit_rules():
    atoms = [
        Atom("A", "LYS", "10", "NZ", "N", (0.0, 3.0, 0.0)),      # valid potential_ionic (3.0 A)
        Atom("A", "HIS", "15", "NE2", "N", (0.0, -3.0, 0.0)),    # conditional His
        Atom("A", "SER", "20", "OG", "O", (2.0, 0.0, 0.0)),      # anomalously short (< 2.4 A)
        Atom("D", "SUB", "1", "O1", "O", (0.0, 0.0, 0.0)),
        Atom("D", "SUB", "1", "O2", "O", (2.2, 1.8, 0.0)),
    ]
    groups = [
        TargetGroup("coo1", "D", "1", ("O1", "O2"), formal_charge=-1, provenance="test_graph")
    ]
    pairs = evaluate_contacts(atoms, groups, ContactPolicy())
    by_res = {(p["res"], p["ri"], p["acceptor_atom"]): p for p in pairs}

    assert by_res[("LYS", "10", "O1")]["accepted"] is True
    assert by_res[("LYS", "10", "O1")]["kind"] == "potential_ionic"
    assert by_res[("HIS", "15", "O1")]["accepted"] is False
    assert by_res[("HIS", "15", "O1")]["conditional"] is True
    assert by_res[("SER", "20", "O1")]["reason"] == "anomalously_short_contact"
    # Whole SER20/coo1 group must be quarantined even if O2 is farther
    assert all(not p["accepted"] for p in pairs if p["ri"] == "20")
    assert distinct_residue_count(pairs) == 1

    collapsed = collapse_for_display(pairs)
    assert len(collapsed) == 3  # LYS10 accepted, HIS15 & SER20 conditional


def test_contact_visualization_report_and_geometry_plot(tmp_path: Path):
    assert build_parser().prog == "contact-analysis"
    p1 = _write_contact_pdb(tmp_path / "pose_0.pdb")
    ens = analyze_ensemble([p1])

    rep = write_contact_report(ens, tmp_path / "report", name="demo", pose_path=p1)
    assert Path(rep["contact_frequency_json"]).is_file()
    assert Path(rep["contact_summary_csv"]).is_file()
    assert Path(rep["contact_report_md"]).is_file()
    if rep["figure"] and "figure_path" in rep["figure"]:
        assert Path(rep["figure"]["figure_path"]).is_file()

    # Test attack-geometry summary plot from pose_geometry.csv
    pose_csv = tmp_path / "pose_geometry.csv"
    with pose_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant_id", "sample", "distance_A", "angle_deg"])
        w.writeheader()
        w.writerow({"variant_id": "WT", "sample": 0, "distance_A": 4.2, "angle_deg": 110.0})
        w.writerow({"variant_id": "D2E", "sample": 0, "distance_A": 3.1, "angle_deg": 168.0})
    rc = ca_cli_main(["plot-geometry", "--pose-geometry-csv", str(pose_csv), "--out", str(tmp_path / "geom.png")])
    assert rc == 0
    assert (tmp_path / "geom.png").is_file()
