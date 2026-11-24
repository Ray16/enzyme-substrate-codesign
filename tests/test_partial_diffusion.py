"""Offline tests for the standalone partial diffusion workflow and CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack_geometry.config import GeometryConfig, LigandSpec, ThresholdConfig
from partial_diffusion.cli import build_parser, main as pd_cli_main
from partial_diffusion.pipeline import (
    MotifResidueSpec,
    PartialDiffusionConfig,
    collect_generated_designs,
    evaluate_partial_diffusion,
    prepare_boltz_refolds,
    run_partial_diffusion_pipeline,
)
from partial_diffusion.runner import (
    AlphaProteinNovoSpec,
    RFD4PartialDiffusionSpec,
    build_apnovo_manifest,
    build_rfd4_command,
    run_diffusion_backend,
)

from ._synthetic import SEQUENCE, write_mock_boltz_prediction, write_synthetic_holo_cif


def test_rfd4_and_apnovo_builders(tmp_path: Path):
    inp = write_synthetic_holo_cif(tmp_path / "motif.cif")
    rfd4_spec = RFD4PartialDiffusionSpec(input_path=str(inp), num_replicates=4, seed=7, partial_t=0.4)
    cmd = build_rfd4_command(rfd4_spec, tmp_path / "out")
    assert cmd[0] == "rfd4-generate"
    assert "num_replicates=4" in cmd
    assert "partial_t=0.4" in cmd

    ap_spec = AlphaProteinNovoSpec(
        motif_cif_path=str(inp),
        partial_diffusion_input_file=str(inp),
        motif_str="A1-5/B1/D1",
        partial_diffusion_num_steps=600,
    )
    manifest = build_apnovo_manifest(ap_spec)
    assert manifest["designs"][0]["partial_diffusion_num_steps"] == 600
    assert manifest["resequence"]["enabled"] is True


def test_end_to_end_partial_diffusion_evaluation(tmp_path: Path):
    designs_dir = tmp_path / "designs"
    write_synthetic_holo_cif(designs_dir / "des_good.cif", seq=SEQUENCE)
    write_synthetic_holo_cif(designs_dir / "des_bad.cif", seq=SEQUENCE)

    designs = collect_generated_designs(designs_dir, protein_chain="A")
    assert len(designs) == 2
    assert all(d.qc_passed for d in designs)

    ligands = [LigandSpec(id="B", ccd="ATP"), LigandSpec(id="D", smiles="O=C(O)CCCCC(=O)O")]
    inputs = prepare_boltz_refolds(designs, ligands, tmp_path)
    assert set(inputs) == {"des_good", "des_bad"}

    results_dir = tmp_path / "boltz_results"
    write_mock_boltz_prediction(
        results_dir, "des_good", SEQUENCE, offsets=[(-3.0, 0.0, 0.0)], confidence_score=0.90, swap_asp_od=True
    )
    write_mock_boltz_prediction(
        results_dir, "des_bad", SEQUENCE, offsets=[(0.0, 4.5, 0.0)], confidence_score=0.75
    )

    geom_cfg = GeometryConfig.from_dict(
        {
            "donor": {"chain": "D", "elements": ["O"], "n_atoms": 2},
            "electrophile": {"chain": "B", "atom": "PA"},
            "leaving": {"chain": "B", "atom": "O3A"},
        }
    )
    motif = [MotifResidueSpec(position=2, resname="ASP", atoms=("CG", "OD1", "OD2"))]
    report = evaluate_partial_diffusion(
        designs=designs,
        results_dir=results_dir,
        geometry_config=geom_cfg,
        output_dir=tmp_path / "eval_out",
        motif_residues=motif,
        thresholds=ThresholdConfig(max_distance_A=4.0, min_angle_deg=120.0, max_motif_rmsd_A=1.5),
    )
    assert report["n_evaluated"] == 2
    assert report["n_passed_thresholds"] == 1
    top = report["top_hits"][0]
    assert top["design_id"] == "des_good"
    assert top["backbone_rmsd_A"] == pytest.approx(0.0, abs=1e-3)
    # Even with swapped ASP OD1/OD2, symmetry-aware motif RMSD is 0.0
    assert top["motif_rmsd_A"] == pytest.approx(0.0, abs=1e-3)
    assert (tmp_path / "eval_out" / "design_summary.csv").is_file()
    assert (tmp_path / "eval_out" / "top_hits.json").is_file()


def test_partial_diffusion_cli_help_and_dry_run(tmp_path: Path):
    parser = build_parser()
    assert parser.prog == "partial-diffusion"

    inp = write_synthetic_holo_cif(tmp_path / "motif.cif")
    cfg = {
        "output_dir": str(tmp_path / "pd_out"),
        "backend": "rfd4",
        "diffusion": {"input_path": str(inp), "num_replicates": 2, "partial_t": 0.5},
        "ligands": [{"id": "B", "ccd": "ATP"}],
    }
    cfg_path = tmp_path / "pd.json"
    cfg_path.write_text(json.dumps(cfg))
    rc = pd_cli_main(["run", "--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    assert (tmp_path / "pd_out" / "diffusion_outputs" / "diffusion_request.json").is_file()
