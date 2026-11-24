"""Offline tests for the standalone saturation mutagenesis workflow and CLI."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from attack_geometry.config import GeometryConfig, LigandSpec, ThresholdConfig
from saturation_mutagenesis.cli import build_parser, main as sat_cli_main
from saturation_mutagenesis.pipeline import (
    SaturationConfig,
    evaluate_saturation_campaign,
    prepare_boltz_inputs,
    run_saturation_pipeline,
)
from saturation_mutagenesis.pocket import define_pocket
from saturation_mutagenesis.variants import build_multisite_variants, build_variants

from ._synthetic import SEQUENCE, write_mock_boltz_prediction, write_synthetic_holo_pdb


def test_pocket_connectivity_admits_bridge_and_excludes_stray(tmp_path: Path):
    pdb = write_synthetic_holo_pdb(tmp_path / "holo.pdb", stray_metal=True, bridging_metal=True)
    pocket = define_pocket(pdb, cutoff=5.0, link_cutoff=5.0, anchor="B", protein_chain="A")
    assert set(pocket.admitted) == {"B", "C", "D"}
    assert [e.component for e in pocket.excluded] == ["Z"]
    assert 2 in pocket.resnums()
    assert 3 in pocket.resnums()  # reached via bridging metal C


def test_single_and_multisite_variant_generation():
    singles = build_variants(SEQUENCE, [(2, "D")], label="enz", alphabet="ADE")
    assert len(singles) == 3  # WT + D2A + D2E
    assert [v.mutation for v in singles] == ["WT", "D2A", "D2E"]

    multis = build_multisite_variants(
        SEQUENCE, [(1, "A"), (2, "D")], label="enz", alphabet="AE", min_mutations=1, max_mutations=2
    )
    muts = {v.mutation for v in multis}
    assert "WT" in muts
    assert "A1E_D2A" in muts
    assert "A1E_D2E" in muts

    explicit = build_multisite_variants(
        SEQUENCE, explicit_mutations=["D2E", "A1G+D2E"], label="enz"
    )
    assert [v.mutation for v in explicit] == ["WT", "D2E", "A1G_D2E"]


def test_end_to_end_saturation_evaluation_and_ranking(tmp_path: Path):
    pdb = write_synthetic_holo_pdb(tmp_path / "holo.pdb")
    variants = build_variants(SEQUENCE, [(2, "D")], label="enz", alphabet="DE")
    ligands = [LigandSpec(id="B", ccd="ATP"), LigandSpec(id="D", smiles="O=C(O)CCCCC(=O)O")]
    inputs = prepare_boltz_inputs(variants, ligands, tmp_path)
    assert set(inputs) == {"enz_WT", "enz_D2E"}

    results_dir = tmp_path / "boltz_results"
    # WT has suboptimal 90 deg angle; D2E has ideal 180 deg inline angle and shorter distance
    write_mock_boltz_prediction(
        results_dir, "enz_WT", SEQUENCE, offsets=[(0.0, 3.8, 0.0), (0.0, 3.6, 0.0)], confidence_score=0.80
    )
    write_mock_boltz_prediction(
        results_dir, "enz_D2E", variants[1].seq, offsets=[(-2.9, 0.0, 0.0), (-3.1, 0.0, 0.0)], confidence_score=0.88
    )

    geom_cfg = GeometryConfig.from_dict(
        {
            "donor": {"chain": "D", "elements": ["O"], "n_atoms": 2},
            "electrophile": {"chain": "B", "atom": "PA"},
            "leaving": {"chain": "B", "atom": "O3A"},
        }
    )
    report = evaluate_saturation_campaign(
        variants=variants,
        results_dir=results_dir,
        geometry_config=geom_cfg,
        output_dir=tmp_path / "eval_out",
        thresholds=ThresholdConfig(max_distance_A=4.0, min_angle_deg=120.0),
        chain_pairs={"prot_sub_iptm": ("0", "2")},
        n_bootstrap=200,
    )
    assert report["n_evaluated"] == 2
    assert report["n_passed_thresholds"] == 1
    assert report["top_hits"][0]["variant_id"] == "enz_D2E"
    assert report["top_hits"][0]["delta_angle_deg"] > 80.0
    assert (tmp_path / "eval_out" / "variant_summary.csv").is_file()
    assert (tmp_path / "eval_out" / "pose_geometry.csv").is_file()
    assert (tmp_path / "eval_out" / "top_hits.json").is_file()


def test_saturation_cli_help_and_dry_run(tmp_path: Path):
    parser = build_parser()
    assert parser.prog == "saturation-mutagenesis"

    pdb = write_synthetic_holo_pdb(tmp_path / "holo.pdb")
    cfg = {
        "structure_path": str(pdb),
        "output_dir": str(tmp_path / "run_out"),
        "positions": [2],
        "alphabet": "DE",
        "ligands": [{"id": "B", "ccd": "ATP"}, {"id": "D", "smiles": "O=C(O)CCCCC(=O)O"}],
    }
    cfg_path = tmp_path / "sat.json"
    cfg_path.write_text(json.dumps(cfg))
    rc = sat_cli_main(["run", "--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    assert (tmp_path / "run_out" / "variants.tsv").is_file()
    assert (tmp_path / "run_out" / "boltz_inputs" / "enzyme_D2E.yaml").is_file()
