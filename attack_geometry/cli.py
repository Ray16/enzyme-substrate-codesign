"""CLI for standalone catalytic attack-geometry evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import GeometryConfig, ThresholdConfig, load_config
from .metrics import clash_fraction, evaluate_attack_geometry, evaluate_thresholds, summarize_ensemble
from .structure import load_structure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzyme-geometry",
        description="Evaluate catalytic nucleophile-electrophile attack geometry on PDB/mmCIF structures.",
    )
    parser.add_argument(
        "structures",
        nargs="+",
        type=Path,
        help="One or more PDB or mmCIF structure files to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML or JSON file containing a 'geometry' section (and optional 'thresholds').",
    )
    parser.add_argument(
        "--protein-chain",
        default="A",
        help="Protein chain ID for clash calculation (default: A).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = load_config(args.config)
    geom_cfg = GeometryConfig.from_dict(raw.get("geometry", raw))
    thresh_cfg = ThresholdConfig.from_dict(raw.get("thresholds"))

    per_pose = []
    geoms = []
    for path in args.structures:
        st = load_structure(path)
        g = evaluate_attack_geometry(st, geom_cfg)
        cf = clash_fraction(st, protein_chain=args.protein_chain, cutoff_A=geom_cfg.clash_cutoff_A)
        metrics = {**g.to_dict(), "clash_fraction": round(cf, 5)}
        passed, reasons = evaluate_thresholds(metrics, thresh_cfg)
        geoms.append(g)
        per_pose.append({"structure": str(path), **metrics, "passed": passed, "fail_reasons": reasons})

    summary = summarize_ensemble(geoms).to_dict()
    ens_passed, ens_reasons = evaluate_thresholds(summary, thresh_cfg)
    payload = {
        "ensemble": {**summary, "passed": ens_passed, "fail_reasons": ens_reasons},
        "poses": per_pose,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
