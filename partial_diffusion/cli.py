"""Command-line interface for the standalone partial diffusion enzyme design workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attack_geometry.boltz import BoltzConfig
from attack_geometry.config import GeometryConfig, LigandSpec, ThresholdConfig, load_config

from .pipeline import (
    MotifResidueSpec,
    PartialDiffusionConfig,
    collect_generated_designs,
    evaluate_partial_diffusion,
    prepare_boltz_refolds,
    run_boltz_refolds,
    run_partial_diffusion_pipeline,
)
from .runner import run_diffusion_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="partial-diffusion",
        description=(
            "Partial diffusion enzyme design workflow: backbone perturbation & "
            "sequence redesign (RFD4-Proteina / AlphaProtein Novo), Boltz-2 "
            "co-folding validation, and catalytic attack-geometry ranking."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. run-diffusion
    p_diff = sub.add_parser("run-diffusion", help="Prepare and run partial diffusion design generation.")
    p_diff.add_argument("--config", type=Path, required=True, help="YAML/JSON config with 'backend' and 'diffusion'.")
    p_diff.add_argument("--output-dir", type=Path, required=True, help="Directory for generated structures.")
    p_diff.add_argument("--dry-run", action="store_true", help="Write manifests/commands without launching GPU inference.")

    # 2. prepare-boltz
    p_prep = sub.add_parser("prepare-boltz", help="Collect generated designs and write Boltz-2 co-folding YAMLs.")
    p_prep.add_argument("--designs-dir", type=Path, required=True, help="Directory containing generated CIF/PDB designs.")
    p_prep.add_argument("--config", type=Path, required=True, help="YAML/JSON config containing 'ligands'.")
    p_prep.add_argument("--protein-chain", default="A", help="Protein chain ID (default: A).")
    p_prep.add_argument("--msa-dir", type=Path, default=None, help="Optional directory of per-design MSA CSV files.")
    p_prep.add_argument("--output-dir", type=Path, required=True, help="Output directory for boltz_inputs/.")

    # 3. run-boltz
    p_boltz = sub.add_parser("run-boltz", help="Run Boltz-2 refolding on prepared design YAMLs.")
    p_boltz.add_argument("--inputs-dir", type=Path, required=True, help="Directory of <design_id>.yaml files.")
    p_boltz.add_argument("--results-dir", type=Path, required=True, help="Directory for Boltz-2 prediction outputs.")
    p_boltz.add_argument("--config", type=Path, default=None, help="Optional YAML/JSON config with a 'boltz' section.")
    p_boltz.add_argument("--diffusion-samples", type=int, default=None, help="Override diffusion sample count.")
    p_boltz.add_argument("--dry-run", action="store_true", help="Print Boltz-2 commands without running.")

    # 4. evaluate
    p_eval = sub.add_parser("evaluate", help="Score Boltz-2 refolds by attack geometry, motif/backbone RMSD, and confidence.")
    p_eval.add_argument("--designs-dir", type=Path, required=True, help="Directory containing generated CIF/PDB designs.")
    p_eval.add_argument("--results-dir", type=Path, required=True, help="Directory containing Boltz-2 refold outputs.")
    p_eval.add_argument("--config", type=Path, required=True, help="YAML/JSON config with 'geometry' and optional 'thresholds'.")
    p_eval.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON summary outputs.")
    p_eval.add_argument("--protein-chain", default="A", help="Protein chain ID (default: A).")
    p_eval.add_argument("--top-k", type=int, default=20, help="Number of top ranked hits in top_hits.json.")
    p_eval.add_argument("--max-distance", type=float, default=None, help="Override max attack distance threshold (Å).")
    p_eval.add_argument("--min-angle", type=float, default=None, help="Override min attack angle threshold (deg).")

    # 5. run (end-to-end)
    p_run = sub.add_parser("run", help="Run end-to-end partial diffusion + Boltz-2 validation from a config.")
    p_run.add_argument("--config", type=Path, required=True, help="Full partial diffusion YAML/JSON config.")
    p_run.add_argument("--skip-diffusion", action="store_true", help="Skip generation and use existing designs_dir.")
    p_run.add_argument("--skip-boltz", action="store_true", help="Skip Boltz-2 execution and score existing outputs.")
    p_run.add_argument("--dry-run", action="store_true", help="Prepare inputs and print commands without GPU execution.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run-diffusion":
        raw = load_config(args.config)
        res = run_diffusion_backend(
            backend=str(raw.get("backend", "rfd4")),
            spec_dict=dict(raw.get("diffusion") or {}),
            out_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return 0

    if args.command == "prepare-boltz":
        raw = load_config(args.config)
        ligands = [LigandSpec.from_dict(x) for x in raw.get("ligands", ())]
        designs = collect_generated_designs(args.designs_dir, protein_chain=args.protein_chain)
        written = prepare_boltz_refolds(
            designs, ligands, args.output_dir, protein_chain=args.protein_chain, msa_dir=args.msa_dir
        )
        sys.stdout.write(
            json.dumps(
                {
                    "n_designs": len(designs),
                    "n_boltz_inputs": len(written),
                    "inputs_dir": str(args.output_dir / "boltz_inputs"),
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    if args.command == "run-boltz":
        raw = load_config(args.config) if args.config else {}
        b_dict = dict(raw.get("boltz") or {})
        if args.diffusion_samples is not None:
            b_dict["diffusion_samples"] = args.diffusion_samples
        b_cfg = BoltzConfig.from_dict(b_dict)
        yamls = {p.stem: p for p in sorted(args.inputs_dir.glob("*.yaml"))}
        runs = run_boltz_refolds(yamls, args.results_dir, b_cfg, dry_run=args.dry_run)
        sys.stdout.write(json.dumps({"n_jobs": len(runs), "runs": runs}, indent=2) + "\n")
        return 0

    if args.command == "evaluate":
        raw = load_config(args.config)
        designs = collect_generated_designs(args.designs_dir, protein_chain=args.protein_chain)
        geom_cfg = GeometryConfig.from_dict(raw["geometry"])
        thresh_dict = dict(raw.get("thresholds") or {})
        if args.max_distance is not None:
            thresh_dict["max_distance_A"] = args.max_distance
        if args.min_angle is not None:
            thresh_dict["min_angle_deg"] = args.min_angle
        thresh_cfg = ThresholdConfig.from_dict(thresh_dict)
        motif = [MotifResidueSpec.from_dict(x) for x in raw.get("motif_residues", ())]
        cp_raw = raw.get("chain_pairs") or {}
        chain_pairs = {str(k): (str(v[0]), str(v[1])) for k, v in cp_raw.items()}
        report = evaluate_partial_diffusion(
            designs=designs,
            results_dir=args.results_dir,
            geometry_config=geom_cfg,
            output_dir=args.output_dir,
            protein_chain=args.protein_chain,
            motif_residues=motif,
            reference_motif_structure=raw.get("reference_motif_structure"),
            thresholds=thresh_cfg,
            chain_pairs=chain_pairs,
            top_k=args.top_k,
        )
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
        return 0

    if args.command == "run":
        cfg = PartialDiffusionConfig.from_file(args.config)
        result = run_partial_diffusion_pipeline(
            cfg,
            run_diffusion=not args.skip_diffusion,
            run_boltz=not args.skip_boltz,
            dry_run=args.dry_run,
        )
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
