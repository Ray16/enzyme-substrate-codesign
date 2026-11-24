"""Command-line interface for standalone Boltz-2 co-folding."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attack_geometry.boltz import BoltzConfig
from attack_geometry.config import LigandSpec, PartnerChainSpec, load_config

from .msa import parse_fasta, prepare_msas_from_fasta
from .predict import (
    CofoldConfig,
    batch_prepare_boltz_inputs,
    run_cofold_batch,
    verify_cofold_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boltz-cofold",
        description="Standalone Boltz-2 co-folding: resumable MSA retrieval, input prep, prediction, and QC.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_msa = sub.add_parser("prepare-msa", help="Fetch and cache sequence-specific MSAs from FASTA.")
    p_msa.add_argument("--fasta", type=Path, required=True, help="Input protein FASTA.")
    p_msa.add_argument("--out-dir", type=Path, required=True, help="Directory for index.tsv.")
    p_msa.add_argument("--cache-dir", type=Path, required=True, help="Content-addressed MSA cache directory.")

    p_prep = sub.add_parser("prepare-inputs", help="Generate Boltz-2 YAML inputs from FASTA and ligand config.")
    p_prep.add_argument("--fasta", type=Path, required=True, help="Input protein FASTA.")
    p_prep.add_argument("--config", type=Path, required=True, help="YAML/JSON config containing 'ligands'.")
    p_prep.add_argument("--protein-chain", default="A", help="Protein chain ID (default: A).")
    p_prep.add_argument("--msa-index", type=Path, default=None, help="Optional index.tsv from prepare-msa.")
    p_prep.add_argument("--output-dir", type=Path, required=True, help="Output directory for boltz_inputs/.")

    p_pred = sub.add_parser("predict", help="Run Boltz-2 co-folding on prepared YAML inputs.")
    p_pred.add_argument("--inputs-dir", type=Path, required=True, help="Directory of <id>.yaml files.")
    p_pred.add_argument("--results-dir", type=Path, required=True, help="Output directory for predictions.")
    p_pred.add_argument("--config", type=Path, default=None, help="Optional YAML/JSON config with 'boltz' section.")
    p_pred.add_argument("--diffusion-samples", type=int, default=None, help="Override diffusion samples.")
    p_pred.add_argument("--fp32", action="store_true", help="Use FP32 precision with serial sample-memory cap.")
    p_pred.add_argument("--dry-run", action="store_true", help="Print commands without executing Boltz-2.")

    p_ver = sub.add_parser("verify", help="Verify cofolded structures, sequences, and confidence JSONs.")
    p_ver.add_argument("--results-dir", type=Path, required=True, help="Directory containing Boltz-2 outputs.")
    p_ver.add_argument("--fasta", type=Path, default=None, help="Optional FASTA to check expected sequences.")
    p_ver.add_argument("--expected-samples", type=int, default=None, help="Expected models per target.")
    p_ver.add_argument("--protein-chain", default="A", help="Protein chain ID (default: A).")
    p_ver.add_argument("--output-dir", type=Path, default=None, help="Optional directory for verification CSV/JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare-msa":
        rows = prepare_msas_from_fasta(args.fasta, args.out_dir, args.cache_dir)
        sys.stdout.write(json.dumps({"n_alignments": len(rows), "index_tsv": str(args.out_dir / "index.tsv")}, indent=2) + "\n")
        return 0

    if args.command == "prepare-inputs":
        seqs = parse_fasta(args.fasta)
        raw = load_config(args.config)
        ligands = [LigandSpec.from_dict(x) for x in raw.get("ligands", ())]
        partners = [PartnerChainSpec.from_dict(x) for x in raw.get("partner_chains", ())]
        written = batch_prepare_boltz_inputs(
            seqs,
            ligands,
            args.output_dir,
            protein_chain=args.protein_chain,
            partner_chains=partners,
            msa_index_tsv=args.msa_index,
        )
        sys.stdout.write(json.dumps({"n_inputs": len(written), "inputs_dir": str(args.output_dir / "boltz_inputs")}, indent=2) + "\n")
        return 0

    if args.command == "predict":
        raw = load_config(args.config) if args.config else {}
        b_dict = dict(raw.get("boltz") or {})
        if args.diffusion_samples is not None:
            b_dict["diffusion_samples"] = args.diffusion_samples
        if args.fp32:
            b_dict["fp32"] = True
        b_cfg = BoltzConfig.from_dict(b_dict)
        yamls = {p.stem: p for p in sorted(args.inputs_dir.glob("*.yaml"))}
        runs = run_cofold_batch(yamls, args.results_dir, b_cfg, dry_run=args.dry_run)
        sys.stdout.write(json.dumps({"n_jobs": len(runs), "runs": runs}, indent=2) + "\n")
        return 0

    if args.command == "verify":
        seqs = parse_fasta(args.fasta) if args.fasta else None
        summary = verify_cofold_outputs(
            args.results_dir,
            expected_sequences=seqs,
            expected_samples=args.expected_samples,
            protein_chain=args.protein_chain,
            output_dir=args.output_dir,
        )
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
