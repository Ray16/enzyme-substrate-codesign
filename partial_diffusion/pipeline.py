"""End-to-end partial diffusion workflow: design -> Boltz-2 refold -> geometry + motif RMSD + ranking."""
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
from attack_geometry.config import GeometryConfig, LigandSpec, ThresholdConfig, load_config
from attack_geometry.metrics import (
    clash_fraction,
    evaluate_attack_geometry,
    evaluate_thresholds,
    kabsch_fit,
    rmsd,
    summarize_ensemble,
    symmetry_rmsd,
)
from attack_geometry.structure import Structure, load_structure

from .runner import run_diffusion_backend, verify_generated_structure

BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass(frozen=True)
class MotifResidueSpec:
    """Catalytic residue specification for motif alignment scoring."""

    position: int                  # 1-based position in designed sequence
    resname: str                   # 3-letter residue name (e.g. "ASP")
    atoms: tuple[str, ...]         # Heavy atoms to align (e.g. ("CG", "OD1", "OD2"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotifResidueSpec":
        return cls(
            position=int(data["position"]),
            resname=str(data["resname"]).upper(),
            atoms=tuple(str(a) for a in data["atoms"]),
        )


@dataclass(frozen=True)
class DesignRecord:
    """One generated partial-diffusion design tracked through Boltz-2 validation."""

    design_id: str
    structure_path: str
    sequence: str
    sequence_sha256: str
    length: int
    qc_passed: bool = True
    qc_error: str = ""


@dataclass
class PartialDiffusionConfig:
    """Complete configuration for a partial diffusion + Boltz-2 validation campaign."""

    output_dir: str
    backend: str = "rfd4"                            # "rfd4" | "apnovo"
    diffusion: dict[str, Any] = field(default_factory=dict)
    designs_dir: str | None = None                   # Optional pre-generated designs dir
    design_glob: str = "**/*.cif"
    protein_chain: str = "A"
    ligands: list[LigandSpec] = field(default_factory=list)
    msa_dir: str | None = None
    boltz: BoltzConfig = field(default_factory=BoltzConfig)
    geometry: GeometryConfig | None = None
    motif_residues: list[MotifResidueSpec] = field(default_factory=list)
    reference_motif_structure: str | None = None
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    chain_pairs: dict[str, tuple[str, str]] = field(default_factory=dict)
    top_k: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartialDiffusionConfig":
        ligands = [LigandSpec.from_dict(x) for x in data.get("ligands", ())]
        geometry = GeometryConfig.from_dict(data["geometry"]) if data.get("geometry") else None
        motif = [MotifResidueSpec.from_dict(x) for x in data.get("motif_residues", ())]
        cp_raw = data.get("chain_pairs") or {}
        chain_pairs = {str(k): (str(v[0]), str(v[1])) for k, v in cp_raw.items()}
        return cls(
            output_dir=str(data.get("output_dir", "partial_diffusion_out")),
            backend=str(data.get("backend", "rfd4")),
            diffusion=dict(data.get("diffusion") or {}),
            designs_dir=str(data["designs_dir"]) if data.get("designs_dir") else None,
            design_glob=str(data.get("design_glob", "**/*.cif")),
            protein_chain=str(data.get("protein_chain", "A")),
            ligands=ligands,
            msa_dir=str(data["msa_dir"]) if data.get("msa_dir") else None,
            boltz=BoltzConfig.from_dict(data.get("boltz")),
            geometry=geometry,
            motif_residues=motif,
            reference_motif_structure=str(data["reference_motif_structure"])
            if data.get("reference_motif_structure")
            else None,
            thresholds=ThresholdConfig.from_dict(data.get("thresholds")),
            chain_pairs=chain_pairs,
            top_k=int(data.get("top_k", 20)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "PartialDiffusionConfig":
        return cls.from_dict(load_config(path))


def collect_generated_designs(
    designs_dir: str | Path,
    *,
    protein_chain: str = "A",
    patterns: tuple[str, ...] = ("*.cif", "*.pdb"),
) -> list[DesignRecord]:
    """Collect and QC-verify all generated structures in ``designs_dir``."""
    root = Path(designs_dir).expanduser()
    if not root.exists():
        return []
    files: list[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    files = sorted({p.resolve() for p in files})

    records: list[DesignRecord] = []
    seen_ids: set[str] = set()
    for p in files:
        base_id = p.stem
        did = base_id
        if did in seen_ids:
            digest = hashlib.sha256(str(p).encode()).hexdigest()[:8]
            did = f"{base_id}_{digest}"
        seen_ids.add(did)
        try:
            info = verify_generated_structure(p, protein_chain=protein_chain)
            seq = info["sequence"]
            records.append(
                DesignRecord(
                    design_id=did,
                    structure_path=str(p),
                    sequence=seq,
                    sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),
                    length=len(seq),
                    qc_passed=True,
                )
            )
        except (ValueError, OSError) as exc:
            records.append(
                DesignRecord(
                    design_id=did,
                    structure_path=str(p),
                    sequence="",
                    sequence_sha256="",
                    length=0,
                    qc_passed=False,
                    qc_error=str(exc),
                )
            )
    return records


def prepare_boltz_refolds(
    designs: Sequence[DesignRecord],
    ligands: Sequence[LigandSpec],
    out_dir: str | Path,
    protein_chain: str = "A",
    msa_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write unrestrained Boltz-2 co-folding inputs for all readable generated designs."""
    inputs_dir = Path(out_dir).expanduser() / "boltz_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    msa_root = Path(msa_dir).expanduser() if msa_dir else None

    written: dict[str, Path] = {}
    for d in designs:
        if not d.sequence:
            continue
        msa_path = None
        if msa_root is not None:
            for cand in (msa_root / f"{d.design_id}.csv", msa_root / f"{d.sequence_sha256[:16]}.csv"):
                if cand.is_file():
                    msa_path = cand
                    break
        doc = build_boltz_input(
            sequence=d.sequence,
            ligands=ligands,
            protein_chain=protein_chain,
            msa_path=msa_path,
        )
        written[d.design_id] = write_boltz_input(doc, inputs_dir / f"{d.design_id}.yaml")
    return written


def run_boltz_refolds(
    input_yamls: dict[str, Path],
    results_dir: str | Path,
    boltz_config: BoltzConfig,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run Boltz-2 refolding predictions for every prepared design YAML."""
    root = Path(results_dir).expanduser()
    records = []
    for did, yaml_path in input_yamls.items():
        res = run_boltz_predict(yaml_path, root / did, boltz_config, dry_run=dry_run)
        records.append({"design_id": did, **res})
    return records


def _backbone_and_motif_rmsd(
    pred: Structure,
    design: Structure,
    protein_chain: str,
    motif_residues: Sequence[MotifResidueSpec],
    ref_motif_structure: Structure | None = None,
) -> dict[str, float]:
    """Compute global N/CA/C/O backbone RMSD and symmetry-aware catalytic motif RMSD."""
    pred_map = pred.atom_map(protein_chain)
    des_map = design.atom_map(protein_chain)
    common_bb = sorted(
        k for k in pred_map if k[1] in BACKBONE_ATOMS and k in des_map
    )
    out: dict[str, float] = {}
    transform = None
    if len(common_bb) >= 3:
        x_bb = np.array([pred_map[k] for k in common_bb], dtype=float)
        y_bb = np.array([des_map[k] for k in common_bb], dtype=float)
        transform = kabsch_fit(x_bb, y_bb)
        out["backbone_rmsd_A"] = round(rmsd(x_bb @ transform[0] + transform[1], y_bb), 4)

    if motif_residues:
        target_map = ref_motif_structure.atom_map(protein_chain) if ref_motif_structure else des_map
        x_m, y_m, desc = [], [], []
        for spec in motif_residues:
            for atom_name in spec.atoms:
                key = (spec.position, atom_name)
                if key in pred_map and key in target_map:
                    x_m.append(pred_map[key])
                    y_m.append(target_map[key])
                    desc.append((spec.position, spec.resname, atom_name))
        if len(x_m) >= 3:
            xm_arr = np.array(x_m, dtype=float)
            ym_arr = np.array(y_m, dtype=float)
            sel_res = symmetry_rmsd(xm_arr, ym_arr, desc)
            out["motif_rmsd_A"] = sel_res["rmsd_A"]
            if transform is not None and ref_motif_structure is None:
                bb_res = symmetry_rmsd(xm_arr, ym_arr, desc, transform=transform)
                out["motif_backbone_aligned_rmsd_A"] = bb_res["rmsd_A"]
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def evaluate_partial_diffusion(
    designs: Sequence[DesignRecord],
    results_dir: str | Path,
    geometry_config: GeometryConfig,
    output_dir: str | Path,
    *,
    protein_chain: str = "A",
    motif_residues: Sequence[MotifResidueSpec] = (),
    reference_motif_structure: str | Path | None = None,
    thresholds: ThresholdConfig | None = None,
    chain_pairs: dict[str, tuple[str, str]] | None = None,
    top_k: int = 20,
    verify_sequence: bool = True,
) -> dict[str, Any]:
    """Score Boltz-2 refolded partial-diffusion designs by geometry, RMSD, clashes, and confidence.

    Outputs:
    - ``design_summary.csv``: ranked per-design ensemble metrics, backbone/motif RMSDs, clashes, confidence
    - ``pose_geometry.csv``: per-pose attack geometry, structural RMSDs, and confidence
    - ``top_hits.json``: machine-readable summary and top-ranked designs
    """
    thresholds = thresholds or ThresholdConfig()
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ref_motif_st = load_structure(reference_motif_structure) if reference_motif_structure else None

    known_ids = {d.design_id for d in designs}
    groups = discover_boltz_models(results_dir, known_ids=known_ids)

    pose_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for d in designs:
        row: dict[str, Any] = {
            "design_id": d.design_id,
            "design_structure": d.structure_path,
            "length": d.length,
            "sequence": d.sequence,
            "generation_qc_passed": d.qc_passed,
            "status": "pending",
            "n_poses": 0,
        }
        refs = groups.get((d.design_id, ""), [])
        if not refs:
            if not d.qc_passed:
                row["status"] = "generation_qc_failed"
                row["error"] = d.qc_error
            summary_rows.append(row)
            continue

        try:
            des_st = load_structure(d.structure_path)
            geoms = []
            confs: dict[str, list[float]] = {}
            clashes: list[float] = []
            rmsds: dict[str, list[float]] = {}
            for ref in refs:
                if not ref.structure_path:
                    raise ValueError(f"Missing structure beside {ref.confidence_path}")
                pred_st = load_structure(ref.structure_path)
                if verify_sequence and d.sequence and pred_st.sequence(protein_chain, strict_numbering=False) != d.sequence:
                    raise ValueError("Refolded sequence differs from generated design")
                g = evaluate_attack_geometry(pred_st, geometry_config)
                cf = clash_fraction(pred_st, protein_chain=protein_chain, cutoff_A=geometry_config.clash_cutoff_A)
                r_metrics = _backbone_and_motif_rmsd(
                    pred_st, des_st, protein_chain, motif_residues, ref_motif_st
                )
                conf = read_confidence(ref.confidence_path, chain_pairs=chain_pairs)
                geoms.append(g)
                clashes.append(cf)
                for k, val in r_metrics.items():
                    rmsds.setdefault(k, []).append(val)
                for k, val in conf.items():
                    if isinstance(val, (int, float)):
                        confs.setdefault(k, []).append(float(val))
                pose_rows.append(
                    {
                        "design_id": d.design_id,
                        "sample": ref.sample,
                        **g.to_dict(),
                        **r_metrics,
                        "clash_fraction": round(cf, 5),
                        **conf,
                    }
                )

            ens = summarize_ensemble(geoms).to_dict()
            rmsd_summary = {
                k: round(statistics.fmean(vals), 4) for k, vals in rmsds.items() if vals
            }
            conf_summary = {
                f"{k}_mean": round(statistics.fmean(vals), 5) for k, vals in confs.items() if vals
            }
            row.update(
                status="evaluated",
                **ens,
                **rmsd_summary,
                clash_fraction=round(statistics.fmean(clashes), 5) if clashes else 0.0,
                **conf_summary,
            )
            passed, reasons = evaluate_thresholds(row, thresholds)
            row["passed_thresholds"] = passed
            row["fail_reasons"] = "; ".join(reasons)
        except (ValueError, OSError) as exc:
            row["status"] = "qc_failed"
            row["error"] = str(exc)

        summary_rows.append(row)

    def _rank_key(r: dict[str, Any]) -> tuple:
        evaluated = r.get("status") == "evaluated"
        passed = bool(r.get("passed_thresholds", False))
        angle = float(r.get("weighted_angle_deg", -1.0)) if evaluated else -1.0
        dist = float(r.get("harmonic_distance_A", 1e9)) if evaluated else 1e9
        conf = float(r.get("confidence_score_mean", -1.0)) if evaluated else -1.0
        return (not passed, not evaluated, -angle, dist, -conf, r["design_id"])

    summary_rows.sort(key=_rank_key)
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank if row.get("status") == "evaluated" else ""

    _write_csv(out / "design_summary.csv", summary_rows)
    if pose_rows:
        _write_csv(out / "pose_geometry.csv", pose_rows)

    evaluated_rows = [r for r in summary_rows if r.get("status") == "evaluated"]
    report = {
        "n_designs": len(designs),
        "n_evaluated": len(evaluated_rows),
        "n_passed_thresholds": sum(1 for r in evaluated_rows if r.get("passed_thresholds")),
        "outputs": {
            "design_summary_csv": str(out / "design_summary.csv"),
            "pose_geometry_csv": str(out / "pose_geometry.csv"),
        },
        "top_hits": evaluated_rows[:top_k],
    }
    (out / "top_hits.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_partial_diffusion_pipeline(
    config: PartialDiffusionConfig,
    *,
    run_diffusion: bool = True,
    run_boltz: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the full partial diffusion + Boltz-2 validation pipeline."""
    out = Path(config.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    diffusion_result = None
    designs_dir = Path(config.designs_dir).expanduser() if config.designs_dir else out / "diffusion_outputs"
    if run_diffusion and config.diffusion:
        diffusion_result = run_diffusion_backend(
            config.backend, config.diffusion, designs_dir, dry_run=dry_run
        )

    designs = collect_generated_designs(designs_dir, protein_chain=config.protein_chain)
    fasta_lines = []
    for d in designs:
        if d.sequence:
            fasta_lines.append(f">{d.design_id}\n{d.sequence}\n")
    if fasta_lines:
        (out / "designs.fasta").write_text("".join(fasta_lines))

    boltz_inputs: dict[str, Path] = {}
    boltz_runs: list[dict[str, Any]] = []
    if designs and config.ligands:
        boltz_inputs = prepare_boltz_refolds(
            designs,
            config.ligands,
            out,
            protein_chain=config.protein_chain,
            msa_dir=config.msa_dir,
        )
        if run_boltz:
            boltz_runs = run_boltz_refolds(
                boltz_inputs, out / "boltz_results", config.boltz, dry_run=dry_run
            )

    eval_report = None
    if designs and config.geometry is not None and not dry_run:
        eval_report = evaluate_partial_diffusion(
            designs=designs,
            results_dir=out / "boltz_results",
            geometry_config=config.geometry,
            output_dir=out,
            protein_chain=config.protein_chain,
            motif_residues=config.motif_residues,
            reference_motif_structure=config.reference_motif_structure,
            thresholds=config.thresholds,
            chain_pairs=config.chain_pairs,
            top_k=config.top_k,
        )

    return {
        "output_dir": str(out),
        "backend": config.backend,
        "diffusion": diffusion_result,
        "n_designs_collected": len(designs),
        "n_boltz_inputs": len(boltz_inputs),
        "boltz_runs": boltz_runs,
        "evaluation": eval_report,
    }
