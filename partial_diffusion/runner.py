"""Partial-diffusion structure perturbation & sequence redesign backend wrappers.

Supports two backends extracted from ``enzyme_design_agent`` without any
cluster-specific paths or job-scheduler dependencies:

1. ``rfd4`` (RFD4-Proteina): all-atom motif/ligand-conditioned partial diffusion
   and sequence-structure co-design via the upstream ``rfd4-generate`` CLI.
2. ``apnovo`` (AlphaProtein Novo): partial-diffusion backbone perturbation from a
   parent structure + LigandMPNN sequence redesign manifest builder.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from attack_geometry.structure import load_structure


@dataclass(frozen=True)
class RFD4PartialDiffusionSpec:
    """Configuration for RFD4-Proteina partial diffusion / motif-conditioned co-design."""

    input_path: str
    executable: str = "rfd4-generate"
    num_replicates: int = 8
    seed: int = 0
    recycle_schedule: str = "midpeak"      # midpeak | front_loaded | flat
    batch_size: int = 1
    num_workers: int = 0
    partial_t: float | None = None         # Optional partial-diffusion noise level (0..1)
    hf_ckpt_path: str = "nvidia/RFD4-Proteina"
    extra_overrides: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RFD4PartialDiffusionSpec":
        return cls(
            input_path=str(data["input_path"]),
            executable=str(data.get("executable", "rfd4-generate")),
            num_replicates=int(data.get("num_replicates", 8)),
            seed=int(data.get("seed", 0)),
            recycle_schedule=str(data.get("recycle_schedule", "midpeak")),
            batch_size=int(data.get("batch_size", 1)),
            num_workers=int(data.get("num_workers", 0)),
            partial_t=float(data["partial_t"]) if data.get("partial_t") is not None else None,
            hf_ckpt_path=str(data.get("hf_ckpt_path", "nvidia/RFD4-Proteina")),
            extra_overrides=tuple(str(x) for x in data.get("extra_overrides", ())),
        )


@dataclass(frozen=True)
class AlphaProteinNovoSpec:
    """Configuration for AlphaProtein Novo partial diffusion + LigandMPNN redesign."""

    motif_cif_path: str
    partial_diffusion_input_file: str
    motif_str: str
    design_name: str = "partial_diff"
    motif_atoms: str = ""
    seq_length: str = ""
    unindexed_motif_residues: str = ""
    num_designs: int = 8
    num_sampling_steps: int = 200
    partial_diffusion_num_steps: int = 600
    seed_start: int = 0
    is_author_naming: bool = True
    resequence: bool = True
    resequence_temperature: float = 0.1
    num_sequences: int = 4
    model_dir: str = ""
    executable: str = "apnovo-run"
    folding_states: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlphaProteinNovoSpec":
        return cls(
            motif_cif_path=str(data["motif_cif_path"]),
            partial_diffusion_input_file=str(data["partial_diffusion_input_file"]),
            motif_str=str(data["motif_str"]),
            design_name=str(data.get("design_name", "partial_diff")),
            motif_atoms=str(data.get("motif_atoms", "")),
            seq_length=str(data.get("seq_length", "")),
            unindexed_motif_residues=str(data.get("unindexed_motif_residues", "")),
            num_designs=int(data.get("num_designs", 8)),
            num_sampling_steps=int(data.get("num_sampling_steps", 200)),
            partial_diffusion_num_steps=int(data.get("partial_diffusion_num_steps", 600)),
            seed_start=int(data.get("seed_start", 0)),
            is_author_naming=bool(data.get("is_author_naming", True)),
            resequence=bool(data.get("resequence", True)),
            resequence_temperature=float(data.get("resequence_temperature", 0.1)),
            num_sequences=int(data.get("num_sequences", 4)),
            model_dir=str(data.get("model_dir", "")),
            executable=str(data.get("executable", "apnovo-run")),
            folding_states=tuple(data.get("folding_states", ())),
        )


def build_rfd4_command(spec: RFD4PartialDiffusionSpec, out_dir: str | Path) -> list[str]:
    """Construct the ``rfd4-generate`` CLI command with explicit Hydra overrides."""
    if spec.recycle_schedule not in ("midpeak", "front_loaded", "flat"):
        raise ValueError(f"Unsupported recycle_schedule: {spec.recycle_schedule!r}")
    out = Path(out_dir).expanduser().resolve()
    inp = Path(spec.input_path).expanduser().resolve()
    cmd = [
        spec.executable,
        "inference_experiment=quickstart",
        f"inputs={inp}",
        f"out_dir={out}",
        f"num_replicates={spec.num_replicates}",
        f"seed={spec.seed}",
        f"recycle_strategy=medium/{spec.recycle_schedule}",
        f"batch_size={spec.batch_size}",
        f"num_workers={spec.num_workers}",
        "prevalidate_inputs=true",
        f"hf_ckpt_path={spec.hf_ckpt_path}",
        "compress_outputs=false",
        f"hydra.run.dir={out / 'hydra'}",
    ]
    if spec.partial_t is not None:
        cmd.append(f"partial_t={spec.partial_t}")
    cmd.extend(spec.extra_overrides)
    return cmd


def build_apnovo_manifest(spec: AlphaProteinNovoSpec) -> dict[str, Any]:
    """Translate ``AlphaProteinNovoSpec`` into an AlphaProtein Novo partial-diffusion manifest."""
    design: dict[str, Any] = {
        "name": spec.design_name,
        "output_prefix": spec.design_name,
        "input_file": str(Path(spec.motif_cif_path).expanduser().resolve()),
        "motif_str": spec.motif_str,
        "num_designs": int(spec.num_designs),
        "is_author_naming": bool(spec.is_author_naming),
        "partial_diffusion_input_file": str(
            Path(spec.partial_diffusion_input_file).expanduser().resolve()
        ),
        "partial_diffusion_num_steps": int(spec.partial_diffusion_num_steps),
    }
    if spec.motif_atoms:
        design["motif_atoms"] = spec.motif_atoms
    if spec.seq_length:
        design["seq_length"] = spec.seq_length
    if spec.unindexed_motif_residues:
        design["unindexed_motif_residues"] = spec.unindexed_motif_residues

    manifest: dict[str, Any] = {
        "defaults": {
            "num_sampling_steps": int(spec.num_sampling_steps),
            "seed_start": int(spec.seed_start),
        },
        "designs": [design],
        "resequence": {"enabled": bool(spec.resequence)},
    }
    if spec.model_dir:
        manifest["settings"] = {"model_dir": str(Path(spec.model_dir).expanduser().resolve())}
    if spec.resequence:
        manifest["resequence"].update(
            {
                "temperature": float(spec.resequence_temperature),
                "num_sequences": int(spec.num_sequences),
                "use_side_chain_context": False,
            }
        )
    if spec.folding_states:
        manifest["folding"] = {
            "inputs": ["resequenced" if spec.resequence else "generated"],
            "states": list(spec.folding_states),
        }
    return manifest


def run_diffusion_backend(
    backend: str,
    spec_dict: dict[str, Any],
    out_dir: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare and optionally execute a partial-diffusion backend job."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if backend == "rfd4":
        spec = RFD4PartialDiffusionSpec.from_dict(spec_dict)
        cmd = build_rfd4_command(spec, out)
        (out / "diffusion_request.json").write_text(
            json.dumps({"backend": "rfd4", "command": cmd}, indent=2) + "\n"
        )
        if dry_run:
            return {"status": "dry_run", "backend": "rfd4", "command": cmd, "out_dir": str(out)}
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "status": "ok" if proc.returncode == 0 else "failed",
            "backend": "rfd4",
            "returncode": proc.returncode,
            "command": cmd,
            "out_dir": str(out),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    if backend == "apnovo":
        spec = AlphaProteinNovoSpec.from_dict(spec_dict)
        manifest = build_apnovo_manifest(spec)
        manifest_path = out / "apnovo_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        cmd = [spec.executable, "--manifest", str(manifest_path), "--output-dir", str(out)]
        if dry_run:
            return {
                "status": "dry_run",
                "backend": "apnovo",
                "manifest_path": str(manifest_path),
                "command": cmd,
                "out_dir": str(out),
            }
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "status": "ok" if proc.returncode == 0 else "failed",
            "backend": "apnovo",
            "returncode": proc.returncode,
            "manifest_path": str(manifest_path),
            "command": cmd,
            "out_dir": str(out),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    raise ValueError(f"Unsupported partial diffusion backend: {backend!r} (expected 'rfd4' or 'apnovo')")


def verify_generated_structure(
    path: str | Path,
    protein_chain: str = "A",
) -> dict[str, Any]:
    """Check finite coordinates and N/CA/C backbone completeness on a generated design."""
    p = Path(path).expanduser()
    st = load_structure(p)
    residues = [r for r in st.residues if r.chain == protein_chain]
    if not residues:
        raise ValueError(f"{p}: no protein residues found in chain {protein_chain!r}")
    for r in residues:
        if not np.isfinite(r.xyz).all():
            raise ValueError(f"{p}: non-finite coordinates in residue {r.chain}{r.resnum}")
        if not {"N", "CA", "C"} <= set(r.atom_names):
            raise ValueError(f"{p}: incomplete backbone (missing N/CA/C) at residue {r.chain}{r.resnum}")
    seq = st.sequence(protein_chain, strict_numbering=False)
    return {
        "structure_path": str(p),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "sequence": seq,
        "length": len(seq),
        "n_atoms": len(st.atoms),
    }
