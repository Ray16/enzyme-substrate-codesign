"""End-to-end saturation mutagenesis workflow: pocket -> variants -> Boltz-2 -> scoring & ranking."""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from attack_geometry.boltz import (
    BoltzConfig,
    build_boltz_input,
    discover_boltz_models,
    read_confidence,
    run_boltz_predict,
    write_boltz_input,
)
from attack_geometry.config import GeometryConfig, LigandSpec, ThresholdConfig, load_config
from attack_geometry.metrics import (
    bootstrap_wt_delta,
    clash_fraction,
    evaluate_attack_geometry,
    evaluate_thresholds,
    summarize_ensemble,
)
from attack_geometry.structure import load_structure

from .pocket import define_pocket
from .variants import (
    AA20,
    WT_MUTATION,
    Variant,
    build_multisite_variants,
    build_variants,
    write_variants_fasta,
    write_variants_tsv,
)


@dataclass
class SaturationConfig:
    """Complete configuration for a saturation mutagenesis campaign."""

    structure_path: str
    output_dir: str
    protein_chain: str = "A"
    label: str = "enzyme"
    positions: list[int] | None = None
    pocket_cutoff_A: float = 5.0
    link_cutoff_A: float = 5.0
    anchor_component: str | None = None
    components: list[str] | None = None
    alphabet: str = AA20
    max_mutations: int = 1
    explicit_mutations: list[str] | None = None
    annotate_positions: dict[int, str] = field(default_factory=dict)
    include_wt: bool = True
    ligands: list[LigandSpec] = field(default_factory=list)
    msa_dir: str | None = None
    boltz: BoltzConfig = field(default_factory=BoltzConfig)
    geometry: GeometryConfig | None = None
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    chain_pairs: dict[str, tuple[str, str]] = field(default_factory=dict)
    n_bootstrap: int = 2000
    seed: int = 0
    top_k: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SaturationConfig":
        ligands = [LigandSpec.from_dict(x) for x in data.get("ligands", ())]
        geometry = GeometryConfig.from_dict(data["geometry"]) if data.get("geometry") else None
        thresholds = ThresholdConfig.from_dict(data.get("thresholds"))
        boltz = BoltzConfig.from_dict(data.get("boltz"))
        ann_raw = data.get("annotate_positions") or {}
        annotate = {int(k): str(v) for k, v in ann_raw.items()}
        cp_raw = data.get("chain_pairs") or {}
        chain_pairs = {str(k): (str(v[0]), str(v[1])) for k, v in cp_raw.items()}
        positions = [int(p) for p in data["positions"]] if data.get("positions") else None
        return cls(
            structure_path=str(data["structure_path"]),
            output_dir=str(data.get("output_dir", "saturation_out")),
            protein_chain=str(data.get("protein_chain", "A")),
            label=str(data.get("label", "enzyme")),
            positions=positions,
            pocket_cutoff_A=float(data.get("pocket_cutoff_A", 5.0)),
            link_cutoff_A=float(data.get("link_cutoff_A", 5.0)),
            anchor_component=data.get("anchor_component"),
            components=list(data["components"]) if data.get("components") else None,
            alphabet=str(data.get("alphabet", AA20)),
            max_mutations=int(data.get("max_mutations", 1)),
            explicit_mutations=list(data["explicit_mutations"]) if data.get("explicit_mutations") else None,
            annotate_positions=annotate,
            include_wt=bool(data.get("include_wt", True)),
            ligands=ligands,
            msa_dir=str(data["msa_dir"]) if data.get("msa_dir") else None,
            boltz=boltz,
            geometry=geometry,
            thresholds=thresholds,
            chain_pairs=chain_pairs,
            n_bootstrap=int(data.get("n_bootstrap", 2000)),
            seed=int(data.get("seed", 0)),
            top_k=int(data.get("top_k", 20)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SaturationConfig":
        return cls.from_dict(load_config(path))


def prepare_boltz_inputs(
    variants: Sequence[Variant],
    ligands: Sequence[LigandSpec],
    out_dir: str | Path,
    protein_chain: str = "A",
    msa_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write one Boltz-2 YAML input per variant into ``out_dir/boltz_inputs``."""
    inputs_dir = Path(out_dir).expanduser() / "boltz_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    msa_root = Path(msa_dir).expanduser() if msa_dir else None

    written: dict[str, Path] = {}
    for v in variants:
        msa_path = None
        if msa_root is not None:
            cand = msa_root / f"{v.id}.csv"
            if cand.is_file():
                msa_path = cand
        doc = build_boltz_input(
            sequence=v.seq,
            ligands=ligands,
            protein_chain=protein_chain,
            msa_path=msa_path,
        )
        written[v.id] = write_boltz_input(doc, inputs_dir / f"{v.id}.yaml")
    return written


def run_boltz_campaign(
    input_yamls: dict[str, Path],
    results_dir: str | Path,
    boltz_config: BoltzConfig,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run Boltz-2 predictions for every prepared variant YAML."""
    root = Path(results_dir).expanduser()
    records = []
    for vid, yaml_path in input_yamls.items():
        res = run_boltz_predict(yaml_path, root / vid, boltz_config, dry_run=dry_run)
        records.append({"variant_id": vid, **res})
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def evaluate_saturation_campaign(
    variants: Sequence[Variant],
    results_dir: str | Path,
    geometry_config: GeometryConfig,
    output_dir: str | Path,
    *,
    protein_chain: str = "A",
    thresholds: ThresholdConfig | None = None,
    chain_pairs: dict[str, tuple[str, str]] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
    top_k: int = 20,
    verify_sequence: bool = True,
) -> dict[str, Any]:
    """Score Boltz-2 predicted structures by catalytic attack geometry and confidence.

    Outputs:
    - ``variant_summary.csv``: ranked per-variant ensemble metrics, WT deltas, confidence, pass/fail
    - ``pose_geometry.csv``: per-pose distance, angle, clash fraction, and confidence
    - ``top_hits.json``: machine-readable campaign summary and top-ranked variants
    """
    thresholds = thresholds or ThresholdConfig()
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    known_ids = {v.id for v in variants}
    groups = discover_boltz_models(results_dir, known_ids=known_ids)

    pose_rows: list[dict[str, Any]] = []
    pose_arrays: dict[str, np.ndarray] = {}
    variant_meta: dict[str, dict[str, Any]] = {}
    variant_errors: dict[str, str] = {}

    for v in variants:
        refs = groups.get((v.id, ""), [])
        if not refs:
            continue
        geoms = []
        confs: dict[str, list[float]] = {}
        clashes: list[float] = []
        try:
            for ref in refs:
                if not ref.structure_path:
                    raise ValueError(f"Missing structure file beside {ref.confidence_path}")
                st = load_structure(ref.structure_path)
                if verify_sequence and st.sequence(protein_chain) != v.seq:
                    raise ValueError("Predicted sequence differs from requested variant")
                g = evaluate_attack_geometry(st, geometry_config)
                cf = clash_fraction(st, protein_chain=protein_chain, cutoff_A=geometry_config.clash_cutoff_A)
                conf = read_confidence(ref.confidence_path, chain_pairs=chain_pairs)
                geoms.append(g)
                clashes.append(cf)
                for k, val in conf.items():
                    if isinstance(val, (int, float)):
                        confs.setdefault(k, []).append(float(val))
                pose_rows.append(
                    {
                        "variant_id": v.id,
                        "mutation": v.mutation,
                        "sample": ref.sample,
                        **g.to_dict(),
                        "clash_fraction": round(cf, 5),
                        **conf,
                    }
                )
            if all(g.angle_deg is not None for g in geoms):
                pose_arrays[v.id] = np.array([[g.distance_A, float(g.angle_deg)] for g in geoms], dtype=float)
            ens = summarize_ensemble(geoms).to_dict()
            conf_summary = {
                f"{k}_mean": round(statistics.fmean(vals), 5)
                for k, vals in confs.items()
                if vals
            }
            variant_meta[v.id] = {
                **ens,
                "clash_fraction": round(statistics.fmean(clashes), 5) if clashes else 0.0,
                **conf_summary,
            }
        except (ValueError, OSError) as exc:
            variant_errors[v.id] = str(exc)

    wt_variants = [v.id for v in variants if v.mutation == WT_MUTATION and v.id in pose_arrays]
    wt_id = wt_variants[0] if wt_variants else None

    summary_rows: list[dict[str, Any]] = []
    for v in variants:
        row: dict[str, Any] = {
            "variant_id": v.id,
            "mutation": v.mutation,
            "kind": v.kind,
            "n_mut": v.n_mut,
            "annotation": v.annotation,
            "sequence": v.seq,
            "status": "pending",
            "n_poses": 0,
        }
        if v.id in variant_errors:
            row["status"] = "qc_failed"
            row["error"] = variant_errors[v.id]
        elif v.id in variant_meta:
            metrics = variant_meta[v.id]
            row.update(status="evaluated", **metrics)
            if wt_id is not None:
                if v.id == wt_id:
                    row.update(
                        delta_angle_deg=0.0,
                        delta_distance_A=0.0,
                        angle_improvement_fraction=0.0,
                        distance_improvement_fraction=0.0,
                    )
                else:
                    row.update(
                        bootstrap_wt_delta(
                            pose_arrays[v.id],
                            pose_arrays[wt_id],
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                    )
            passed, reasons = evaluate_thresholds(row, thresholds)
            row["passed_thresholds"] = passed
            row["fail_reasons"] = "; ".join(reasons)
        summary_rows.append(row)

    # Rank: passed first, then evaluated, then larger weighted attack angle, then shorter harmonic distance
    def _rank_key(r: dict[str, Any]) -> tuple:
        evaluated = r.get("status") == "evaluated"
        passed = bool(r.get("passed_thresholds", False))
        angle = float(r.get("weighted_angle_deg", -1.0)) if evaluated else -1.0
        dist = float(r.get("harmonic_distance_A", 1e9)) if evaluated else 1e9
        conf = float(r.get("confidence_score_mean", -1.0)) if evaluated else -1.0
        return (not passed, not evaluated, -angle, dist, -conf, r["variant_id"])

    summary_rows.sort(key=_rank_key)
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank if row.get("status") == "evaluated" else ""

    _write_csv(out / "variant_summary.csv", summary_rows)
    if pose_rows:
        _write_csv(out / "pose_geometry.csv", pose_rows)

    evaluated_rows = [r for r in summary_rows if r.get("status") == "evaluated"]
    top_hits = evaluated_rows[:top_k]
    report = {
        "n_variants": len(variants),
        "n_evaluated": len(evaluated_rows),
        "n_passed_thresholds": sum(1 for r in evaluated_rows if r.get("passed_thresholds")),
        "n_qc_failed": len(variant_errors),
        "wt_reference_id": wt_id,
        "outputs": {
            "variant_summary_csv": str(out / "variant_summary.csv"),
            "pose_geometry_csv": str(out / "pose_geometry.csv"),
        },
        "top_hits": top_hits,
    }
    (out / "top_hits.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_saturation_pipeline(
    config: SaturationConfig,
    *,
    run_boltz: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the full saturation mutagenesis pipeline."""
    out = Path(config.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    st = load_structure(config.structure_path)
    seq = st.sequence(config.protein_chain)

    pocket_rows: list[dict[str, Any]] = []
    if config.positions is not None:
        pos_list = [(p, seq[p - 1]) for p in sorted(set(config.positions))]
    else:
        pocket = define_pocket(
            st,
            cutoff=config.pocket_cutoff_A,
            components=config.components,
            anchor=config.anchor_component,
            link_cutoff=config.link_cutoff_A,
            protein_chain=config.protein_chain,
        )
        pocket_rows = pocket.to_rows()
        if pocket_rows:
            _write_csv(out / "pocket_positions.csv", pocket_rows)
        pos_list = [(p.resnum, p.wt_aa) for p in pocket.positions]

    ann_fn = (lambda rn, _wt: config.annotate_positions.get(rn, "")) if config.annotate_positions else None
    if config.explicit_mutations is not None or config.max_mutations > 1:
        variants = build_multisite_variants(
            seq,
            pos_list if config.explicit_mutations is None else None,
            label=config.label,
            alphabet=config.alphabet,
            max_mutations=config.max_mutations,
            explicit_mutations=config.explicit_mutations,
            annotate=ann_fn,
            include_wt=config.include_wt,
        )
    else:
        variants = build_variants(
            seq,
            pos_list,
            label=config.label,
            alphabet=config.alphabet,
            annotate=ann_fn,
            include_wt=config.include_wt,
        )

    tsv_path = write_variants_tsv(variants, out / "variants.tsv")
    fasta_path = write_variants_fasta(variants, out / "variants.fasta")

    boltz_inputs: dict[str, Path] = {}
    boltz_runs: list[dict[str, Any]] = []
    if config.ligands:
        boltz_inputs = prepare_boltz_inputs(
            variants,
            config.ligands,
            out,
            protein_chain=config.protein_chain,
            msa_dir=config.msa_dir,
        )
        if run_boltz:
            boltz_runs = run_boltz_campaign(
                boltz_inputs, out / "boltz_results", config.boltz, dry_run=dry_run
            )

    eval_report = None
    if config.geometry is not None and not dry_run:
        eval_report = evaluate_saturation_campaign(
            variants=variants,
            results_dir=out / "boltz_results",
            geometry_config=config.geometry,
            output_dir=out,
            protein_chain=config.protein_chain,
            thresholds=config.thresholds,
            chain_pairs=config.chain_pairs,
            n_bootstrap=config.n_bootstrap,
            seed=config.seed,
            top_k=config.top_k,
        )

    return {
        "output_dir": str(out),
        "n_positions": len(pos_list),
        "n_variants": len(variants),
        "variants_tsv": str(tsv_path),
        "variants_fasta": str(fasta_path),
        "n_boltz_inputs": len(boltz_inputs),
        "boltz_runs": boltz_runs,
        "evaluation": eval_report,
    }
