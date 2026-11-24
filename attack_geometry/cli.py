"""CLI for standalone mechanism-aware catalytic geometry evaluation."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .config import GeometryConfig, load_config
from .mechanisms import get_mechanism, list_mechanisms, merge_mechanism_thresholds
from .metrics import (
    clash_fraction,
    evaluate_attack_geometry,
    evaluate_extra_metrics,
    evaluate_extra_thresholds,
    evaluate_thresholds,
    summarize_ensemble,
)
from .structure import load_structure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzyme-geometry",
        description="Evaluate mechanism-aware catalytic geometry on PDB/mmCIF structures.",
    )
    parser.add_argument(
        "structures",
        nargs="*",
        type=Path,
        help="One or more PDB or mmCIF structure files to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML or JSON file containing 'geometry' (and optional 'mechanism' / 'thresholds').",
    )
    parser.add_argument(
        "--mechanism",
        default=None,
        help="Optional reaction mechanism preset (e.g., inline_sn2, carbonyl_addition, general_acid_base).",
    )
    parser.add_argument(
        "--list-mechanisms",
        action="store_true",
        help="Print built-in reaction mechanism specifications and exit.",
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
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_mechanisms:
        sys.stdout.write(json.dumps(list_mechanisms(), indent=2) + "\n")
        return 0

    if not args.structures or args.config is None:
        parser.error("Provide one or more structures and --config (or pass --list-mechanisms).")

    raw = load_config(args.config)
    mech = get_mechanism(args.mechanism or raw.get("mechanism"))
    geom_cfg = GeometryConfig.from_dict(raw.get("geometry", raw))
    thresh_cfg = merge_mechanism_thresholds(mech, raw.get("thresholds"))

    per_pose = []
    geoms = []
    extra_accum: dict[str, list[float]] = {}
    for path in args.structures:
        st = load_structure(path)
        g = evaluate_attack_geometry(st, geom_cfg)
        extras = evaluate_extra_metrics(st, geom_cfg)
        cf = clash_fraction(st, protein_chain=args.protein_chain, cutoff_A=geom_cfg.clash_cutoff_A)
        metrics = {**g.to_dict(), **extras, "clash_fraction": round(cf, 5)}
        passed, reasons = evaluate_thresholds(metrics, thresh_cfg)
        extra_reasons = evaluate_extra_thresholds(metrics, geom_cfg)
        all_reasons = reasons + extra_reasons
        geoms.append(g)
        for k, v in extras.items():
            extra_accum.setdefault(k, []).append(v)
        per_pose.append(
            {
                "structure": str(path),
                **metrics,
                "passed": passed and not extra_reasons,
                "fail_reasons": all_reasons,
            }
        )

    summary = summarize_ensemble(geoms).to_dict()
    for k, vals in extra_accum.items():
        summary[k] = round(statistics.fmean(vals), 4)
    ens_passed, ens_reasons = evaluate_thresholds(summary, thresh_cfg)
    ens_extra = evaluate_extra_thresholds(summary, geom_cfg)
    payload = {
        "mechanism": mech.name,
        "target_angle_deg": mech.target_angle_deg,
        "ensemble": {
            **summary,
            "passed": ens_passed and not ens_extra,
            "fail_reasons": ens_reasons + ens_extra,
        },
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
