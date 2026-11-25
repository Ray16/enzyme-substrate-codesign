"""Boltz-2 co-folding runner (standard & FP32 serial-capped profile) and post-fold QC.

Includes the guarded process-local sample-batching patch from
``structural_modeling_agent/cofold/boltz_serial_predict.py`` for GPUs without
native bfloat16 support or tight VRAM budgets.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import re
import statistics
import subprocess
import sys
import textwrap
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
from attack_geometry.config import LigandSpec, NucleicAcidSpec, PartnerChainSpec, load_config
from attack_geometry.structure import load_structure

from .msa import parse_fasta


@dataclass
class CofoldConfig:
    """Configuration for a standalone Boltz-2 co-folding batch."""

    output_dir: str
    fasta_path: str | None = None
    protein_chain: str = "A"
    partner_chains: list[PartnerChainSpec] = field(default_factory=list)
    nucleic_acids: list[NucleicAcidSpec] = field(default_factory=list)
    protein_modifications: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    ligands: list[LigandSpec] = field(default_factory=list)
    msa_index_tsv: str | None = None
    boltz: BoltzConfig = field(default_factory=BoltzConfig)
    chain_pairs: dict[str, tuple[str, str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CofoldConfig":
        ligands = [LigandSpec.from_dict(x) for x in data.get("ligands", ())]
        partners = [PartnerChainSpec.from_dict(x) for x in data.get("partner_chains", ())]
        nas = [NucleicAcidSpec.from_dict(x) for x in data.get("nucleic_acids", ())]
        mods = [dict(x) for x in data.get("protein_modifications", ())]
        constraints = [dict(x) for x in data.get("constraints", ())]
        cp_raw = data.get("chain_pairs") or {}
        chain_pairs = {str(k): (str(v[0]), str(v[1])) for k, v in cp_raw.items()}
        return cls(
            output_dir=str(data.get("output_dir", "boltz_cofold_out")),
            fasta_path=str(data["fasta_path"]) if data.get("fasta_path") else None,
            protein_chain=str(data.get("protein_chain", "A")),
            partner_chains=partners,
            nucleic_acids=nas,
            protein_modifications=mods,
            constraints=constraints,
            ligands=ligands,
            msa_index_tsv=str(data["msa_index_tsv"]) if data.get("msa_index_tsv") else None,
            boltz=BoltzConfig.from_dict(data.get("boltz")),
            chain_pairs=chain_pairs,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "CofoldConfig":
        return cls.from_dict(load_config(path))


def serial_sampler_source(source: str) -> tuple[str, bool]:
    """Replace Boltz-2's ``sample_ids.chunk(multiplicity % max_parallel_samples + 1)``
    with ``sample_ids.split(max_parallel_samples)`` so ``--max_parallel_samples 1``
    strictly bounds peak VRAM.
    """
    pattern = r"sample_ids\.chunk\(\s*multiplicity % max_parallel_samples \+ 1\s*\)"
    if len(re.findall(pattern, source)) == 1:
        return re.sub(pattern, "sample_ids.split(max_parallel_samples)", source), True
    if "sample_ids.split(max_parallel_samples)" in source:
        return source, False
    raise RuntimeError("Unrecognized Boltz diffusion batching implementation; no patch applied")


def patch_boltz_fp32_serial() -> dict[str, Any]:
    """Apply the process-local FP32 + serial sample-cap patch to an imported ``boltz`` runtime."""
    import boltz.main as engine
    from boltz.model.modules import diffusionv2

    source = textwrap.dedent(inspect.getsource(diffusionv2.AtomDiffusion.sample))
    fixed, changed = serial_sampler_source(source)
    if changed:
        ns: dict[str, Any] = {}
        exec(compile(fixed, __file__, "exec"), diffusionv2.__dict__, ns)
        diffusionv2.AtomDiffusion.sample = ns["sample"]

    original_trainer = engine.Trainer

    def fp32_trainer(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("precision") not in (None, 32, "32-true", "bf16-mixed"):
            raise RuntimeError(f"Unrecognized requested precision: {kwargs.get('precision')!r}")
        kwargs["precision"] = "32-true"
        return original_trainer(*args, **kwargs)

    engine.Trainer = fp32_trainer
    return {
        "producer": "boltz_serial_fp32_v1",
        "precision": "32-true",
        "sample_batch_patch": changed,
        "original_sampler_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "executed_sampler_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
    }


def batch_prepare_boltz_inputs(
    sequences: dict[str, str],
    ligands: Sequence[LigandSpec],
    out_dir: str | Path,
    *,
    protein_chain: str = "A",
    partner_chains: Sequence[PartnerChainSpec] = (),
    nucleic_acids: Sequence[NucleicAcidSpec] = (),
    protein_modifications: Sequence[dict[str, Any]] = (),
    constraints: Sequence[dict[str, Any]] = (),
    msa_index_tsv: str | Path | None = None,
) -> dict[str, Path]:
    """Write one Boltz-2 YAML input per sequence in ``sequences``."""
    inputs_dir = Path(out_dir).expanduser() / "boltz_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    msa_map: dict[str, str] = {}
    if msa_index_tsv is not None:
        with Path(msa_index_tsv).expanduser().open(newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("id") and row.get("csv"):
                    msa_map[row["id"]] = row["csv"]
                if row.get("seqhash") and row.get("csv"):
                    msa_map[row["seqhash"]] = row["csv"]

    written: dict[str, Path] = {}
    for ident, seq in sequences.items():
        seq_hash = hashlib.sha256(seq.encode("utf-8")).hexdigest()
        doc = build_boltz_input(
            sequence=seq,
            ligands=ligands,
            protein_chain=protein_chain,
            msa_path=msa_map.get(ident) or msa_map.get(seq_hash),
            partner_chains=partner_chains,
            nucleic_acids=nucleic_acids,
            protein_modifications=protein_modifications,
            constraints=constraints,
        )
        written[ident] = write_boltz_input(doc, inputs_dir / f"{ident}.yaml")
    return written


def run_cofold_batch(
    input_yamls: dict[str, Path],
    results_dir: str | Path,
    boltz_config: BoltzConfig,
    *,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Run Boltz-2 predictions (standard or FP32 serial-capped) for all YAML inputs."""
    root = Path(results_dir).expanduser()
    records: list[dict[str, Any]] = []
    for ident, yaml_path in input_yamls.items():
        res = run_boltz_predict(
            yaml_path,
            root / ident,
            boltz_config,
            dry_run=dry_run,
            skip_existing=skip_existing,
        )
        records.append({"id": ident, **res})
    return records


def verify_cofold_outputs(
    results_dir: str | Path,
    *,
    expected_sequences: dict[str, str] | None = None,
    expected_samples: int | None = None,
    protein_chain: str = "A",
    chain_pairs: dict[str, tuple[str, str]] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify cofolded Boltz-2 outputs (sequence identity, finite coords, confidence JSONs)."""
    known = set(expected_sequences) if expected_sequences else None
    groups = discover_boltz_models(results_dir, known_ids=known)
    all_ids = sorted(known | {k[0] for k in groups}) if known else sorted({k[0] for k in groups})

    rows: list[dict[str, Any]] = []
    for ident in all_ids:
        refs = groups.get((ident, ""), [])
        row: dict[str, Any] = {
            "id": ident,
            "n_models": len(refs),
            "complete": (len(refs) >= expected_samples) if expected_samples else (len(refs) > 0),
            "status": "missing" if not refs else "verified",
            "errors": [],
        }
        confs: dict[str, list[float]] = {}
        for ref in refs:
            if not ref.structure_path:
                row["errors"].append(f"model_{ref.sample}: missing structure")
                continue
            try:
                st = load_structure(ref.structure_path)
                if not all(np.all(np.isfinite(r.xyz)) for r in st.residues):
                    raise ValueError("non-finite coordinates")
                if expected_sequences and ident in expected_sequences:
                    if st.sequence(protein_chain, strict_numbering=False) != expected_sequences[ident]:
                        raise ValueError("predicted sequence differs from expected")
                c = read_confidence(ref.confidence_path, chain_pairs=chain_pairs)
                for k, v in c.items():
                    if isinstance(v, (int, float)):
                        confs.setdefault(k, []).append(float(v))
            except (ValueError, OSError) as exc:
                row["errors"].append(f"model_{ref.sample}: {exc}")
        if row["errors"]:
            row["status"] = "qc_failed"
            row["complete"] = False
        for k, vals in confs.items():
            row[f"{k}_mean"] = round(statistics.fmean(vals), 5)
        row["error_summary"] = "; ".join(row.pop("errors"))
        rows.append(row)

    summary = {
        "n_targets": len(all_ids),
        "n_verified": sum(1 for r in rows if r["status"] == "verified"),
        "n_complete": sum(1 for r in rows if r["complete"]),
        "n_qc_failed": sum(1 for r in rows if r["status"] == "qc_failed"),
        "targets": rows,
    }
    if output_dir is not None:
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "cofold_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with (out / "cofold_verification.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    return summary
