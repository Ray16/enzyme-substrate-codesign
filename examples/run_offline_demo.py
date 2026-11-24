"""Self-contained offline demo of all five modules (no GPU required).

Run:
    python examples/run_offline_demo.py --out-dir demo_out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from a source checkout without installing
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack_geometry.config import GeometryConfig, LigandSpec, PartnerChainSpec
from attack_geometry.mechanisms import list_mechanisms
from boltz_cofold.predict import verify_cofold_outputs
from contact_analysis.substrate_contacts import analyze_ensemble
from contact_analysis.visualize import plot_attack_geometry_summary, write_contact_report
from saturation_mutagenesis.pipeline import evaluate_saturation_campaign, prepare_boltz_inputs
from saturation_mutagenesis.pocket import define_pocket
from saturation_mutagenesis.variants import build_variants
from tests._synthetic import SEQUENCE, write_mock_boltz_prediction, write_synthetic_holo_pdb


def run_demo(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    holo_pdb = write_synthetic_holo_pdb(out_dir / "scaffold_holo.pdb")

    # 1. Whole-complex pocket scan
    pocket = define_pocket(holo_pdb, cutoff=5.0, link_cutoff=5.0, anchor="B", protein_chain="A")

    # 2. Saturation variant library + oligomeric Boltz-2 input preparation
    variants = build_variants(SEQUENCE, [(2, "D")], label="demo", alphabet="DE")
    ligands = [LigandSpec(id="B", ccd="ATP"), LigandSpec(id="D", smiles="O=C(O)CCCCC(=O)O")]
    partners = [PartnerChainSpec(id="E", sequence=SEQUENCE)]
    inputs = prepare_boltz_inputs(variants, ligands, out_dir, partner_chains=partners)

    # 3. Mock Boltz-2 cofold predictions + verification
    results_dir = out_dir / "boltz_results"
    write_mock_boltz_prediction(
        results_dir, "demo_WT", SEQUENCE, offsets=[(0.0, 3.6, 0.0), (0.0, 3.8, 0.0)], confidence_score=0.80
    )
    write_mock_boltz_prediction(
        results_dir, "demo_D2E", variants[1].seq, offsets=[(-3.0, 0.0, 0.0), (-3.1, 0.0, 0.0)], confidence_score=0.90
    )
    qc = verify_cofold_outputs(results_dir, expected_sequences={v.id: v.seq for v in variants}, expected_samples=2)

    # 4. Mechanism-aware evaluation (inline_sn2)
    geom_cfg = GeometryConfig.from_dict(
        {
            "donor": {"chain": "D", "elements": ["O"], "n_atoms": 2},
            "electrophile": {"chain": "B", "atom": "PA"},
            "leaving": {"chain": "B", "atom": "O3A"},
        }
    )
    eval_report = evaluate_saturation_campaign(
        variants=variants,
        results_dir=results_dir,
        geometry_config=geom_cfg,
        output_dir=out_dir / "evaluation",
        mechanism="inline_sn2",
        n_bootstrap=200,
    )

    # 5. Active-site contact analysis + figures
    pose_files = sorted((results_dir / "demo_D2E").rglob("*.pdb"))
    freq = analyze_ensemble(pose_files)
    contact_rep = write_contact_report(freq, out_dir / "contacts", name="demo_D2E", pose_path=pose_files[0])
    geom_plot = None
    try:
        geom_plot = plot_attack_geometry_summary(
            out_dir / "evaluation" / "pose_geometry.csv",
            out_dir / "evaluation" / "attack_geometry_summary.png",
        )
    except ImportError:
        pass

    return {
        "available_mechanisms": [m["name"] for m in list_mechanisms()],
        "pocket_positions": pocket.resnums(),
        "n_boltz_inputs": len(inputs),
        "cofold_qc_verified": qc["n_verified"],
        "top_variant": eval_report["top_hits"][0]["variant_id"],
        "contact_report": contact_rep,
        "geometry_plot": geom_plot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("demo_out"))
    args = parser.parse_args()
    summary = run_demo(args.out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
