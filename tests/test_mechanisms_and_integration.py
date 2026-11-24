"""Tests for mechanism-aware ranking, extra_metrics, multi-chain Boltz inputs, MSA index handoff, and offline demo."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack_geometry.cli import main as geom_cli_main
from attack_geometry.config import GeometryConfig, LigandSpec, PartnerChainSpec
from attack_geometry.mechanisms import get_mechanism, list_mechanisms
from attack_geometry.metrics import evaluate_extra_metrics, ranking_sort_key
from contact_analysis.visualize import _ligand_pymol_selections
from saturation_mutagenesis.pipeline import evaluate_saturation_campaign, prepare_boltz_inputs
from saturation_mutagenesis.variants import build_variants

from ._synthetic import SEQUENCE, write_mock_boltz_prediction, write_synthetic_holo_pdb


def test_mechanism_registry_and_burgi_dunitz_ranking(tmp_path: Path):
    names = {m["name"] for m in list_mechanisms()}
    assert {"inline_sn2", "carbonyl_addition", "general_acid_base", "metalloenzyme_coordination", "custom"} <= names

    assert geom_cli_main(["--list-mechanisms"]) == 0

    carb = get_mechanism("carbonyl_addition")
    assert carb.target_angle_deg == pytest.approx(107.0)

    # Under carbonyl_addition (target=107°), a 108° pose must rank ABOVE a 180° pose!
    variants = build_variants(SEQUENCE, [(2, "D")], label="hyd", alphabet="DE")
    results_dir = tmp_path / "boltz_results"
    # WT: angle ~180 deg (wrong for carbonyl addition)
    write_mock_boltz_prediction(results_dir, "hyd_WT", SEQUENCE, offsets=[(-3.0, 0.0, 0.0)], confidence_score=0.85)
    # D2E: angle ~108 deg (cos(108 deg) ≈ -0.309, sin(108 deg) ≈ 0.951) -> ideal Bürgi-Dunitz
    write_mock_boltz_prediction(
        results_dir, "hyd_D2E", variants[1].seq, offsets=[(-0.93, 2.85, 0.0)], confidence_score=0.85
    )

    geom_cfg = GeometryConfig.from_dict(
        {
            "donor": {"chain": "D", "elements": ["O"], "n_atoms": 2},
            "electrophile": {"chain": "B", "atom": "PA"},
            "leaving": {"chain": "B", "atom": "O3A"},
            "extra_metrics": [
                {
                    "name": "metal_bridge",
                    "atom_a": {"chain": "C", "atom": "MG"},
                    "atom_b": {"chain": "B", "atom": "O3A"},
                    "max_distance_A": 5.0,
                }
            ],
        }
    )
    report = evaluate_saturation_campaign(
        variants=variants,
        results_dir=results_dir,
        geometry_config=geom_cfg,
        output_dir=tmp_path / "eval_carb",
        mechanism="carbonyl_addition",
        n_bootstrap=100,
    )
    assert report["mechanism"] == "carbonyl_addition"
    assert report["target_angle_deg"] == pytest.approx(107.0)
    assert report["top_hits"][0]["variant_id"] == "hyd_D2E"
    assert report["top_hits"][0]["passed_thresholds"] is True
    assert "metal_bridge_distance_A" in report["top_hits"][0]


def test_distance_only_geometry_and_extra_metrics(tmp_path: Path):
    pdb = write_synthetic_holo_pdb(tmp_path / "holo.pdb", donor_offset=(-3.0, 0.0, 0.0))
    cfg = GeometryConfig.from_dict(
        {
            "donor": {"chain": "D", "elements": ["O"], "n_atoms": 2},
            "electrophile": {"chain": "B", "atom": "PA"},
            "extra_metrics": [
                {
                    "name": "metal_to_leaving",
                    "atom_a": {"chain": "C", "atom": "MG"},
                    "atom_b": {"chain": "B", "atom": "O3A"},
                }
            ],
        }
    )
    extras = evaluate_extra_metrics(pdb, cfg)
    assert extras["metal_to_leaving_distance_A"] > 0.0


def test_partner_chains_and_msa_index_handoff(tmp_path: Path):
    variants = build_variants(SEQUENCE, [(2, "D")], label="enz", alphabet="DE")
    msa_csv = tmp_path / "dummy_msa.csv"
    msa_csv.write_text(f"key,sequence\n-1,{SEQUENCE}\n")
    idx_tsv = tmp_path / "index.tsv"
    idx_tsv.write_text(f"id\tseqhash\tcsv\nenz_WT\tabc\t{msa_csv}\n")

    ligands = [LigandSpec(id="B", ccd="ATP")]
    partners = [PartnerChainSpec(id="E", sequence=SEQUENCE)]
    inputs = prepare_boltz_inputs(
        variants, ligands, tmp_path, msa_index_tsv=idx_tsv, partner_chains=partners
    )
    wt_doc = json.loads(inputs["enz_WT"].read_text())
    proteins = [e["protein"] for e in wt_doc["sequences"] if "protein" in e]
    assert [p["id"] for p in proteins] == ["A", "E"]
    assert proteins[0]["msa"] == str(msa_csv.resolve())


def test_general_pymol_selection_builder(tmp_path: Path):
    pdb = write_synthetic_holo_pdb(tmp_path / "holo.pdb", stray_metal=False)
    cof_sel, metal_sel, sub_sel = _ligand_pymol_selections(pdb)
    assert "chain B" in cof_sel
    assert "chain C" in metal_sel
    assert "chain D" in sub_sel
