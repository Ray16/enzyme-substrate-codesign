"""End-to-end saturation mutagenesis workflow: pocket -> variants -> Boltz-2 -> mechanism-aware scoring."""
from __future__ import annotations

import csv
import hashlib
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
from attack_geometry.config import (
    GeometryConfig,
    LigandSpec,
    NucleicAcidSpec,
    PartnerChainSpec,
    ThresholdConfig,
    load_config,
)
from attack_geometry.mechanisms import get_mechanism, merge_mechanism_thresholds
from attack_geometry.metrics import (
    bootstrap_wt_delta,
    clash_fraction,
    evaluate_attack_geometry,
    evaluate_extra_metrics,
    evaluate_extra_thresholds,
    evaluate_thresholds,
    ranking_sort_key,
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
    partner_chains: list[PartnerChainSpec] = field(default_factory=list)
    nucleic_acids: list[NucleicAcidSpec] = field(default_factory=list)
    protein_modifications: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    label: str = "enzyme"
    mechanism: str = "custom"
    target_angle_deg: float | None = None
    rank_by: str | None = None
    positions: list[int] | None = None
    pocket_cutoff_A: float = 5.0
    link_cutoff_A: float = 5.0
    anchor_component: str | None = None
    components: list[str] | None = None
    alphabet: str = AA20
    min_mutations: int = 1
    max_mutations: int = 1
    max_variants: int | None = None
    explicit_mutations: list[str] | None = None
    annotate_positions: dict[int, str] = field(default_factory=dict)
    include_wt: bool = True
    strict_numbering: bool = False
    skip_existing: bool = True
    ligands: list[LigandSpec] = field(default_factory=list)
    msa_dir: str | None = None
    msa_index_tsv: str | None = None
    boltz: BoltzConfig = field(default_factory=BoltzConfig)
    geometry: GeometryConfig | None = None
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    chain_pairs: dict[str, tuple[str, str]] = field(default_factory=dict)
    n_bootstrap: int = 2000
    seed: int = 0
    top_k: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SaturationConfig":
        mech = get_mechanism(data.get("mechanism"))
        ligands = [LigandSpec.from_dict(x) for x in data.get("ligands", ())]
        partners = [PartnerChainSpec.from_dict(x) for x in data.get("partner_chains", ())]
        nas = [NucleicAcidSpec.from_dict(x) for x in data.get("nucleic_acids", ())]
        mods = [dict(x) for x in data.get("protein_modifications", ())]
        constraints = [dict(x) for x in data.get("constraints", ())]
        geometry = GeometryConfig.from_dict(data["geometry"]) if data.get("geometry") else None
        thresholds = merge_mechanism_thresholds(mech, data.get("thresholds"))
        boltz = BoltzConfig.from_dict(data.get("boltz"))
        ann_raw = data.get("annotate_positions") or {}
        annotate = {int(k): str(v) for k, v in ann_raw.items()}
        cp_raw = data.get("chain_pairs") or {}
        chain_pairs = {str(k): (str(v[0]), str(v[1])) for k, v in cp_raw.items()}
        positions = [int(p) for p in data["positions"]] if data.get("positions") else None
        target_angle = (
            float(data["target_angle_deg"])
            if data.get("target_angle_deg") is not None
            else mech.target_angle_deg
        )
        rank_by = str(data["rank_by"]) if data.get("rank_by") else mech.default_rank_by
        return cls(
            structure_path=str(data["structure_path"]),
            output_dir=str(data.get("output_dir", "saturation_out")),
            protein_chain=str(data.get("protein_chain", "A")),
            partner_chains=partners,
            nucleic_acids=nas,
            protein_modifications=mods,
            constraints=constraints,
            label=str(data.get("label", "enzyme")),
            mechanism=mech.name,
            target_angle_deg=target_angle,
            rank_by=rank_by,
            positions=positions,
            pocket_cutoff_A=float(data.get("pocket_cutoff_A", 5.0)),
            link_cutoff_A=float(data.get("link_cutoff_A", 5.0)),
            anchor_component=data.get("anchor_component"),
            components=list(data["components"]) if data.get("components") else None,
            alphabet=str(data.get("alphabet", AA20)),
            min_mutations=int(data.get("min_mutations", 1)),
            max_mutations=int(data.get("max_mutations", 1)),
            max_variants=int(data["max_variants"]) if data.get("max_variants") is not None else None,
            explicit_mutations=list(data["explicit_mutations"]) if data.get("explicit_mutations") else None,
            annotate_positions=annotate,
            include_wt=bool(data.get("include_wt", True)),
            strict_numbering=bool(data.get("strict_numbering", False)),
            skip_existing=bool(data.get("skip_existing", True)),
            ligands=ligands,
            msa_dir=str(data["msa_dir"]) if data.get("msa_dir") else None,
            msa_index_tsv=str(data["msa_index_tsv"]) if data.get("msa_index_tsv") else None,
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


def _load_msa_index(msa_index_tsv: str | Path | None) -> dict[str, str]:
    if msa_index_tsv is None:
        return {}
    mapping: dict[str, str] = {}
    with Path(msa_index_tsv).expanduser().open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("id") and row.get("csv"):
                mapping[row["id"]] = row["csv"]
            if row.get("seqhash") and row.get("csv"):
                mapping[row["seqhash"]] = row["csv"]
    return mapping


def prepare_boltz_inputs(
    variants: Sequence[Variant],
    ligands: Sequence[LigandSpec],
    out_dir: str | Path,
    protein_chain: str = "A",
    msa_dir: str | Path | None = None,
    msa_index_tsv: str | Path | None = None,
    partner_chains: Sequence[PartnerChainSpec] = (),
    nucleic_acids: Sequence[NucleicAcidSpec] = (),
    protein_modifications: Sequence[dict[str, Any]] = (),
    constraints: Sequence[dict[str, Any]] = (),
) -> dict[str, Path]:
    """Write one Boltz-2 YAML input per variant into ``out_dir/boltz_inputs``."""
    inputs_dir = Path(out_dir).expanduser() / "boltz_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    msa_root = Path(msa_dir).expanduser() if msa_dir else None
    msa_index = _load_msa_index(msa_index_tsv)

    written: dict[str, Path] = {}
    for v in variants:
        seq_hash = hashlib.sha256(v.seq.encode("utf-8")).hexdigest()
        msa_path: str | Path | None = msa_index.get(v.id) or msa_index.get(seq_hash)
        if msa_path is None and msa_root is not None:
            for cand in (msa_root / f"{v.id}.csv", msa_root / f"{seq_hash}.csv"):
                if cand.is_file():
                    msa_path = cand
                    break
        doc = build_boltz_input(
            sequence=v.seq,
            ligands=ligands,
            protein_chain=protein_chain,
            msa_path=msa_path,
            partner_chains=partner_chains,
            nucleic_acids=nucleic_acids,
            protein_modifications=protein_modifications,
            constraints=constraints,
        )
        written[v.id] = write_boltz_input(doc, inputs_dir / f"{v.id}.yaml")
    return written


def run_boltz_campaign(
    input_yamls: dict[str, Path],
    results_dir: str | Path,
    boltz_config: BoltzConfig,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Run Boltz-2 predictions (standard or FP32) for every prepared variant YAML."""
    root = Path(results_dir).expanduser()
    records = []
    for vid, yaml_path in input_yamls.items():
        res = run_boltz_predict(
            yaml_path,
            root / vid,
            boltz_config,
            dry_run=dry_run,
            skip_existing=skip_existing,
        )
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
    mechanism: str = "custom",
    target_angle_deg: float | None = None,
    rank_by: str | None = None,
    thresholds: ThresholdConfig | None = None,
    chain_pairs: dict[str, tuple[str, str]] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
    top_k: int = 20,
    verify_sequence: bool = True,
) -> dict[str, Any]:
    """Score Boltz-2 predicted structures by mechanism-aware catalytic geometry and confidence."""
    mech = get_mechanism(mechanism)
    eff_target_angle = target_angle_deg if target_angle_deg is not None else mech.target_angle_deg
    eff_rank_by = rank_by or mech.default_rank_by
    eff_thresholds = thresholds or mech.default_thresholds

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
        extras_accum: dict[str, list[float]] = {}
        try:
            for ref in refs:
                if not ref.structure_path:
                    raise ValueError(f"Missing structure file beside {ref.confidence_path}")
                st = load_structure(ref.structure_path)
                if verify_sequence and st.sequence(protein_chain, strict_numbering=False) != v.seq:
                    raise ValueError("Predicted sequence differs from requested variant")
                g = evaluate_attack_geometry(st, geometry_config)
                extras = evaluate_extra_metrics(st, geometry_config)
                cf = clash_fraction(st, protein_chain=None, cutoff_A=geometry_config.clash_cutoff_A)
                conf = read_confidence(ref.confidence_path, chain_pairs=chain_pairs)
                geoms.append(g)
                clashes.append(cf)
                for k, val in extras.items():
                    extras_accum.setdefault(k, []).append(val)
                for k, val in conf.items():
                    if isinstance(val, (int, float)):
                        confs.setdefault(k, []).append(float(val))
                pose_rows.append(
                    {
                        "variant_id": v.id,
                        "mutation": v.mutation,
                        "sample": ref.sample,
                        **g.to_dict(),
                        **extras,
                        "clash_fraction": round(cf, 5),
                        **conf,
                    }
                )
            if all(g.angle_deg is not None for g in geoms):
                pose_arrays[v.id] = np.array([[g.distance_A, float(g.angle_deg)] for g in geoms], dtype=float)
            else:
                pose_arrays[v.id] = np.array([g.distance_A for g in geoms], dtype=float)
            ens = summarize_ensemble(geoms).to_dict()
            extra_summary = {
                k: round(statistics.fmean(vals), 4) for k, vals in extras_accum.items() if vals
            }
            conf_summary = {
                f"{k}_mean": round(statistics.fmean(vals), 5)
                for k, vals in confs.items()
                if vals
            }
            variant_meta[v.id] = {
                **ens,
                **extra_summary,
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
            if wt_id is not None and v.id in pose_arrays:
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
                            target_angle_deg=eff_target_angle,
                        )
                    )
            passed, reasons = evaluate_thresholds(row, eff_thresholds)
            extra_reasons = evaluate_extra_thresholds(row, geometry_config)
            all_reasons = reasons + extra_reasons
            row["passed_thresholds"] = passed and not extra_reasons
            row["fail_reasons"] = "; ".join(all_reasons)
        summary_rows.append(row)

    summary_rows.sort(
        key=lambda r: ranking_sort_key(
            r,
            id_key="variant_id",
            rank_by=eff_rank_by,
            target_angle_deg=eff_target_angle,
        )
    )
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank if row.get("status") == "evaluated" else ""

    _write_csv(out / "variant_summary.csv", summary_rows)
    if pose_rows:
        _write_csv(out / "pose_geometry.csv", pose_rows)

    evaluated_rows = [r for r in summary_rows if r.get("status") == "evaluated"]
    top_hits = evaluated_rows[:top_k]
    report = {
        "mechanism": mech.name,
        "target_angle_deg": eff_target_angle,
        "rank_by": eff_rank_by,
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
    seq = st.sequence(config.protein_chain, strict_numbering=config.strict_numbering)
    res_map = st.residue_index_map(config.protein_chain)

    def _to_seq_pos(pdb_resnum: int) -> tuple[int, str]:
        if pdb_resnum in res_map:
            return res_map[pdb_resnum]
        if 1 <= pdb_resnum <= len(seq):
            return pdb_resnum, seq[pdb_resnum - 1]
        raise ValueError(
            f"Position {pdb_resnum} not found in chain {config.protein_chain!r} "
            f"(length {len(seq)})"
        )

    pocket_rows: list[dict[str, Any]] = []
    if config.positions is not None:
        pos_list = [_to_seq_pos(p) for p in sorted(set(config.positions))]
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
        pos_list = [_to_seq_pos(p.resnum) for p in pocket.positions]

    ann_fn = (lambda rn, _wt: config.annotate_positions.get(rn, "")) if config.annotate_positions else None
    if config.explicit_mutations is not None or config.max_mutations > 1:
        variants = build_multisite_variants(
            seq,
            pos_list if config.explicit_mutations is None else None,
            label=config.label,
            alphabet=config.alphabet,
            min_mutations=config.min_mutations,
            max_mutations=config.max_mutations,
            max_variants=config.max_variants,
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
    if config.ligands or config.nucleic_acids or config.partner_chains:
        boltz_inputs = prepare_boltz_inputs(
            variants,
            config.ligands,
            out,
            protein_chain=config.protein_chain,
            msa_dir=config.msa_dir,
            msa_index_tsv=config.msa_index_tsv,
            partner_chains=config.partner_chains,
            nucleic_acids=config.nucleic_acids,
            protein_modifications=config.protein_modifications,
            constraints=config.constraints,
        )
        if run_boltz:
            boltz_runs = run_boltz_campaign(
                boltz_inputs,
                out / "boltz_results",
                config.boltz,
                dry_run=dry_run,
                skip_existing=config.skip_existing,
            )

    eval_report = None
    if config.geometry is not None and not dry_run:
        eval_report = evaluate_saturation_campaign(
            variants=variants,
            results_dir=out / "boltz_results",
            geometry_config=config.geometry,
            output_dir=out,
            protein_chain=config.protein_chain,
            mechanism=config.mechanism,
            target_angle_deg=config.target_angle_deg,
            rank_by=config.rank_by,
            thresholds=config.thresholds,
            chain_pairs=config.chain_pairs,
            n_bootstrap=config.n_bootstrap,
            seed=config.seed,
            top_k=config.top_k,
        )

    return {
        "output_dir": str(out),
        "mechanism": config.mechanism,
        "n_positions": len(pos_list),
        "n_variants": len(variants),
        "variants_tsv": str(tsv_path),
        "variants_fasta": str(fasta_path),
        "n_boltz_inputs": len(boltz_inputs),
        "boltz_runs": boltz_runs,
        "evaluation": eval_report,
    }
