"""Regression & generality tests across diverse enzyme classes.

Covers:
1. Pure-Python mmCIF parsing with unquoted ribose/nucleotide prime atoms (C1', O2', O5').
2. Real-world experimental PDB numbering (N-terminal offset + missing internal loop).
3. Symmetric multi-electrophile substrates and 4-atom signed dihedrals (Re/Si prochiral face).
4. Mechanism-aware bootstrap_wt_delta (Bürgi-Dunitz target angle & distance-only mechanisms).
5. Homodimer sequence propagation (homomer=True), covalent constraints, PTMs, and DNA/RNA chains.
6. Multi-chain oligomer clash screening & multi-chain catalytic motif RMSD.
7. Unified FP32 command generation and skip_existing resumability across pipelines.
8. Chemical perception in contact_analysis (ester vs. carboxylate, phosphorylated substrate,
   heme cofactor, backbone oxyanion-hole heavy-atom audit, distance-only & angle-window plots).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from attack_geometry.boltz import BoltzConfig, build_boltz_command, build_boltz_input, run_boltz_predict
from attack_geometry.config import (
    GeometryConfig,
    LigandSpec,
    NucleicAcidSpec,
    PartnerChainSpec,
    ThresholdConfig,
)
from attack_geometry.mechanisms import list_mechanisms
from attack_geometry.metrics import (
    bootstrap_wt_delta,
    clash_fraction,
    evaluate_attack_geometry,
    evaluate_thresholds,
    summarize_ensemble,
)
from attack_geometry.structure import load_structure
from contact_analysis.heavy_atom_audit import Atom, ContactPolicy, TargetGroup, evaluate_contacts
from contact_analysis.substrate_contacts import analyze_pose, detect_functional_groups, split_ligands
from contact_analysis.visualize import plot_attack_geometry_summary
from partial_diffusion.pipeline import MotifResidueSpec, collect_generated_designs, evaluate_partial_diffusion
from saturation_mutagenesis.pipeline import (
    SaturationConfig,
    evaluate_saturation_campaign,
    prepare_boltz_inputs,
    run_saturation_pipeline,
)
from saturation_mutagenesis.variants import build_variants

from ._synthetic import SEQUENCE, write_mock_boltz_prediction, write_synthetic_holo_cif


def test_mmcif_unquoted_ribose_prime_atoms(tmp_path: Path):
    """Unquoted C1', O2', O5' atom names (ATP/NAD/FAD/CoA) must parse cleanly."""
    cif = tmp_path / "nad_ribose.cif"
    cif.write_text(
        "\n".join(
            [
                "data_nad",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.type_symbol",
                "ATOM CA ALA A 1 0.0 0.0 0.0 C",
                "HETATM C1' NAD B 1 5.0 0.0 0.0 C",
                "HETATM O2' NAD B 1 6.2 0.0 0.0 O",
                "HETATM O5' NAD B 1 5.0 1.4 0.0 O",
                "HETATM C4N NAD B 1 8.0 0.0 0.0 C",
            ]
        )
        + "\n"
    )
    st = load_structure(cif)
    names = {a.name for a in st.atoms if a.chain == "B"}
    assert {"C1'", "O2'", "O5'", "C4N"} == names


def test_non_1_based_and_gapped_pdb_in_saturation_pipeline(tmp_path: Path):
    """Experimental crystal structures starting at resnum 45 with a missing loop work out of the box."""
    pdb = tmp_path / "crystal_offset.pdb"
    lines = []
    atom_id = 1
    # Residues 45 (ALA), 46 (ASP), gap, 50 (SER)
    for idx, (rn, resn) in enumerate([(45, "ALA"), (46, "ASP"), (50, "SER")]):
        x = idx * 4.0
        for aname, dx, dy, elem in [("N", 0.0, 0.0, "N"), ("CA", 1.2, 0.0, "C"), ("C", 2.4, 0.0, "C"), ("O", 2.4, -1.1, "O")]:
            lines.append(
                f"ATOM  {atom_id:5d}  {aname:<3s} {resn:>3s} A{rn:4d}    {x+dx:8.3f}{dy:8.3f}{0.0:8.3f}  1.00 90.00          {elem:>2s}"
            )
            atom_id += 1
    lines.append(
        f"HETATM{atom_id:5d}  C1  LIG L   1       4.500   2.500   0.000  1.00 90.00           C"
    )
    lines.append("END")
    pdb.write_text("\n".join(lines) + "\n")

    cfg = SaturationConfig.from_dict(
        {
            "structure_path": str(pdb),
            "output_dir": str(tmp_path / "sat_offset_out"),
            "positions": [46],  # PDB resnum 46 -> sequential index 2 (ASP)
            "alphabet": "DE",
            "ligands": [{"id": "L", "smiles": "CC(=O)OC"}],
        }
    )
    res = run_saturation_pipeline(cfg, run_boltz=True, dry_run=True)
    assert res["n_variants"] == 2  # WT + D2E
    tsv_text = Path(res["variants_tsv"]).read_text()
    assert "D2E" in tsv_text
    assert "AES" in tsv_text


def test_symmetric_electrophile_and_prochiral_dihedral(tmp_path: Path):
    """Multi-candidate electrophile picks closest center, and 4-atom signed dihedral distinguishes Re/Si face."""
    pdb = tmp_path / "prochiral.pdb"
    # Donor on chain A above (+z) vs below (-z) a planar ketone C1(=O1)-C2 on chain L
    lines = [
        "ATOM      1  OG  SER A  10       0.000   0.000   3.000  1.00 90.00           O",
        "HETATM    2  C1  SUB L   1       0.000   0.000   0.000  1.00 90.00           C",
        "HETATM    3  O1  SUB L   1       1.200   0.000   0.000  1.00 90.00           O",
        "HETATM    4  C2  SUB L   1      -0.600   1.200   0.000  1.00 90.00           C",
        "HETATM    5  C3  SUB L   1       9.000   9.000   0.000  1.00 90.00           C",
        "END",
    ]
    pdb.write_text("\n".join(lines) + "\n")

    cfg = GeometryConfig.from_dict(
        {
            "donor": {"chain": "A", "resnum": 10, "atom": "OG"},
            "electrophile": {"chain": "L", "atom_names": ["C1", "C3"]},
            "leaving": {"chain": "L", "atom": "O1"},
            "dihedral_ref": {"chain": "L", "atom": "C2"},
        }
    )
    g = evaluate_attack_geometry(pdb, cfg)
    assert g.center_atom == "C1"
    assert g.distance_A == pytest.approx(3.0, abs=1e-4)
    assert g.angle_deg == pytest.approx(90.0, abs=1e-3)
    assert g.dihedral_deg is not None
    assert abs(g.dihedral_deg) == pytest.approx(90.0, abs=1.0)

    ens = summarize_ensemble([g])
    assert ens.mean_dihedral_deg == pytest.approx(g.dihedral_deg, abs=1e-2)
    passed, _ = evaluate_thresholds(
        ens.to_dict(), ThresholdConfig(max_distance_A=4.0, min_angle_deg=80.0, min_dihedral_deg=-120.0, max_dihedral_deg=120.0)
    )
    assert passed is True


def test_mechanism_aware_bootstrap_carbonyl_and_distance_only():
    """bootstrap_wt_delta respects target_angle_deg=107 and supports distance-only arrays."""
    # WT at 145 deg (38 deg off 107°); variant at 108 deg (1 deg off 107°)
    wt = np.array([[3.2, 145.0], [3.3, 146.0], [3.1, 144.0]])
    var = np.array([[3.0, 108.0], [3.1, 107.0], [2.9, 109.0]])
    res = bootstrap_wt_delta(var, wt, n_bootstrap=200, seed=0, target_angle_deg=107.0)
    assert res["delta_angle_error_deg"] < -30.0
    assert res["angle_improvement_fraction"] == pytest.approx(1.0)
    assert res["distance_improvement_fraction"] > 0.9

    # Distance-only (general_acid_base / metalloenzyme_coordination)
    wt_d = np.array([3.8, 3.9, 4.0])
    var_d = np.array([2.4, 2.5, 2.6])
    res_d = bootstrap_wt_delta(var_d, wt_d, n_bootstrap=200, seed=0)
    assert res_d["delta_distance_A"] < -1.0
    assert res_d["distance_improvement_fraction"] == pytest.approx(1.0)
    assert "delta_angle_deg" not in res_d


def test_homodimer_propagation_covalent_constraints_and_nucleic_acids(tmp_path: Path):
    """homomer=True propagates mutated sequence & MSA to partner chain; supports constraints/PTMs/DNA."""
    variants = build_variants(SEQUENCE, [(2, "D")], label="dimer", alphabet="DE")
    msa_wt = tmp_path / "dimer_WT.csv"
    msa_mut = tmp_path / "dimer_D2E.csv"
    msa_wt.write_text(f"key,sequence\n-1,{SEQUENCE}\n")
    msa_mut.write_text(f"key,sequence\n-1,{variants[1].seq}\n")

    partners = [PartnerChainSpec(id="B", homomer=True)]
    nas = [NucleicAcidSpec(id="N", kind="dna", sequence="ACGTACGT")]
    ligands = [LigandSpec(id="L", ccd="PLP")]
    mods = [{"position": 4, "ccd": "SEP"}]
    constraints = [{"bond": {"atom1": ["A", 2, "NZ"], "atom2": ["L", 1, "C4A"]}}]

    inputs = prepare_boltz_inputs(
        variants,
        ligands,
        tmp_path,
        msa_dir=tmp_path,
        partner_chains=partners,
        nucleic_acids=nas,
        protein_modifications=mods,
        constraints=constraints,
    )
    mut_doc = json.loads(inputs["dimer_D2E"].read_text())
    proteins = [e["protein"] for e in mut_doc["sequences"] if "protein" in e]
    assert [p["id"] for p in proteins] == ["A", "B"]
    assert proteins[0]["sequence"] == variants[1].seq
    assert proteins[1]["sequence"] == variants[1].seq  # Propagated!
    assert proteins[1]["msa"] == str(msa_mut.resolve())
    assert proteins[0]["modifications"] == mods
    assert any("dna" in e for e in mut_doc["sequences"])
    assert mut_doc["constraints"] == constraints


def test_multichain_clash_and_unified_fp32_skip_existing(tmp_path: Path):
    """clash_fraction catches partner-chain clashes; fp32 & skip_existing work in run_boltz_predict."""
    pdb = tmp_path / "dimer_clash.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C",
                "ATOM      2  CA  ALA B   1      10.000   0.000   0.000  1.00 90.00           C",
                "HETATM    3  C1  LIG L   1      10.500   0.000   0.000  1.00 90.00           C",
                "END",
            ]
        )
        + "\n"
    )
    st = load_structure(pdb)
    assert clash_fraction(st, protein_chain="A", cutoff_A=1.5) == 0.0
    assert clash_fraction(st, protein_chain=None, cutoff_A=1.5) > 0.0

    # Unified FP32 command builder
    yaml_p = tmp_path / "cand1.yaml"
    yaml_p.write_text("{}\n")
    fp32_cfg = BoltzConfig(fp32=True, diffusion_samples=1)
    cmd = build_boltz_command(yaml_p, tmp_path / "res", fp32_cfg)
    assert "boltz_cofold.fp32_worker" in cmd

    # skip_existing resumes without re-running completed predictions
    res_dir = tmp_path / "res"
    write_mock_boltz_prediction(res_dir, "cand1", SEQUENCE, offsets=[(-3.0, 0.0, 0.0)])
    rec = run_boltz_predict(yaml_p, res_dir, fp32_cfg, dry_run=False, skip_existing=True)
    assert rec["status"] == "skipped_existing"


def test_contact_analysis_ester_phosphorylated_substrate_heme_and_oxyanion(tmp_path: Path):
    """Esters are neutral (not carboxylate), single phospho-substrates aren't cofactors, heme is cofactor."""
    # 1. Ester substrate + Asp H-bond acceptor (neutral ester must NOT be charge=-1)
    ester_pdb = tmp_path / "ester.pdb"
    ester_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A  50       0.000   3.000   0.000  1.00 90.00           N",
                "ATOM      2  CA  GLY A  50      -1.200   3.500   0.000  1.00 90.00           C",
                "ATOM      3  OD1 ASP A  60       0.000  -3.100   0.000  1.00 90.00           O",
                # Methyl acetate ester: C1(=O1)-O2-C2
                "HETATM    4  C1  EST L   1       0.000   0.000   0.000  1.00 90.00           C",
                "HETATM    5  O1  EST L   1       0.000   1.220   0.000  1.00 90.00           O",
                "HETATM    6  O2  EST L   1       1.300  -0.300   0.000  1.00 90.00           O",
                "HETATM    7  C2  EST L   1       2.200   0.800   0.000  1.00 90.00           C",
                "END",
            ]
        )
        + "\n"
    )
    pose = analyze_pose(ester_pdb)
    kinds = {g["kind"]: g["charge"] for g in pose["groups"]}
    assert kinds.get("ester") == 0
    assert "carboxylate" not in kinds
    # Asp60 OD1 should be allowed to H-bond to neutral ester (not suppressed as anion-anion repulsion)
    ester_grp = [g for g in pose["groups"] if g["kind"] == "ester"][0]
    cats = {c["res"]: c["category"] for c in ester_grp["contacts"]}
    assert cats.get("GLY50") == "backbone_hbond"
    assert cats.get("ASP60") == "h_bond"

    # 2. Single phosphorylated substrate (e.g. phosphatase / kinase substrate) stays substrate
    g6p_pdb = tmp_path / "g6p.pdb"
    g6p_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1      20.000   0.000   0.000  1.00 90.00           C",
                "HETATM    2  P1  G6P L   1       0.000   0.000   0.000  1.00 90.00           P",
                "HETATM    3  O1  G6P L   1       1.500   0.000   0.000  1.00 90.00           O",
                "HETATM    4  O2  G6P L   1      -1.500   0.000   0.000  1.00 90.00           O",
                "HETATM    5  O3  G6P L   1       0.000   1.500   0.000  1.00 90.00           O",
                "HETATM    6  O4  G6P L   1       0.000  -1.500   0.000  1.00 90.00           O",
                "HETATM    7  C1  G6P L   1       0.000  -2.900   0.000  1.00 90.00           C",
                "HETATM    8  C2  G6P L   1       1.400  -3.400   0.000  1.00 90.00           C",
                "END",
            ]
        )
        + "\n"
    )
    cof, _met, sub = split_ligands(load_structure(g6p_pdb))
    assert len(cof) == 0
    assert len(sub) == 7

    # 3. Heme (non-phosphorus cofactor) is auto-classified as cofactor
    heme_pdb = tmp_path / "p450.pdb"
    heme_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  CYS A   1       0.000   0.000   0.000  1.00 90.00           C",
                "HETATM    2  FE  HEM H   1       5.000   0.000   0.000  1.00 90.00          FE",
                "HETATM    3  NA  HEM H   1       7.000   0.000   0.000  1.00 90.00           N",
                "HETATM    4  C1  SUB L   1       5.000   3.500   0.000  1.00 90.00           C",
                "END",
            ]
        )
        + "\n"
    )
    cof_h, _met_h, sub_h = split_ligands(load_structure(heme_pdb))
    assert len(cof_h) == 2
    assert len(sub_h) == 1

    # 4. Backbone oxyanion-hole audit & multi-chain support in heavy_atom_audit
    atoms = [
        Atom("A", "GLY", "50", "N", "N", (0.0, 2.9, 0.0)),   # oxyanion hole backbone N
        Atom("B", "TYR", "12", "OH", "O", (0.0, -3.0, 0.0)), # partner chain B H-bond
        Atom("L", "EST", "1", "O1", "O", (0.0, 0.0, 0.0)),
    ]
    groups = [TargetGroup("carbonyl_o", "L", "1", ("O1",), formal_charge=0, provenance="ester")]
    pol = ContactPolicy(scope="protein_all_to_ligand_acceptor")
    pairs = evaluate_contacts(atoms, groups, policy=pol, protein_chain=["A", "B"])
    accepted = {(p["chain"], p["res"], p["atom"]): p["kind"] for p in pairs if p["accepted"]}
    assert accepted[("A", "GLY", "N")] == "backbone_amide_polar"
    assert accepted[("B", "TYR", "OH")] == "donor_acceptor_polar"


def test_plot_geometry_distance_only_and_angle_window(tmp_path: Path):
    """plot_attack_geometry_summary handles distance-only CSVs and Bürgi-Dunitz angle windows."""
    # Distance-only CSV (empty angle_deg)
    dist_csv = tmp_path / "dist_only.csv"
    with dist_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant_id", "sample", "distance_A", "angle_deg"])
        w.writeheader()
        w.writerow({"variant_id": "WT", "sample": 0, "distance_A": 4.2, "angle_deg": ""})
        w.writerow({"variant_id": "H10A", "sample": 0, "distance_A": 2.5, "angle_deg": ""})
    res1 = plot_attack_geometry_summary(dist_csv, tmp_path / "dist_only.png", near_distance_A=3.0)
    assert Path(res1["figure_path"]).is_file()
    assert res1["top_candidate"]["id"] == "H10A"

    # Bürgi-Dunitz angle window (85..130, target 107): 108 deg must beat 175 deg
    bd_csv = tmp_path / "bd.csv"
    with bd_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant_id", "sample", "distance_A", "angle_deg"])
        w.writeheader()
        w.writerow({"variant_id": "inline_bad", "sample": 0, "distance_A": 3.0, "angle_deg": 175.0})
        w.writerow({"variant_id": "bd_good", "sample": 0, "distance_A": 3.0, "angle_deg": 108.0})
    res2 = plot_attack_geometry_summary(
        bd_csv,
        tmp_path / "bd.png",
        near_distance_A=3.8,
        min_angle_deg=85.0,
        max_angle_deg=130.0,
        target_angle_deg=107.0,
    )
    assert Path(res2["figure_path"]).is_file()
    assert res2["top_candidate"]["id"] == "bd_good"


def test_design_glob_filtering_and_new_mechanisms(tmp_path: Path):
    """design_glob filters collected designs and new mechanism presets are registered."""
    names = {m["name"] for m in list_mechanisms()}
    assert {"hydride_transfer", "covalent_catalysis"} <= names

    sub = tmp_path / "designs"
    write_synthetic_holo_cif(sub / "keep_1.cif")
    write_synthetic_holo_cif(sub / "skip_1.cif")
    kept = collect_generated_designs(sub, design_glob="keep_*.cif")
    assert [d.design_id for d in kept] == ["keep_1"]
