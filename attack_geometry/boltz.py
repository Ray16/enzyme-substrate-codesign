"""Boltz-2 input preparation, prediction runner, model discovery, and confidence parsing."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import LigandSpec

CONFIDENCE_RE = re.compile(r"^confidence_(?P<id>.+)_model_(?P<sample>\d+)\.json$")

DEFAULT_CONFIDENCE_SCALARS: tuple[str, ...] = (
    "iptm",
    "ligand_iptm",
    "complex_plddt",
    "complex_iplddt",
    "confidence_score",
)


@dataclass(frozen=True)
class BoltzConfig:
    """Parameters for running Boltz-2 co-folding predictions."""

    executable: str = "boltz"
    cache_dir: str | None = None
    diffusion_samples: int = 5
    recycling_steps: int = 3
    sampling_steps: int = 200
    max_parallel_samples: int = 1
    output_format: str = "pdb"
    accelerator: str = "gpu"
    devices: int = 1
    num_workers: int = 0
    preprocessing_threads: int = 1
    seed: int = 0
    use_msa_server: bool = False
    fp32: bool = False
    extra_args: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BoltzConfig":
        if not data:
            return cls()
        return cls(
            executable=str(data.get("executable", "boltz")),
            cache_dir=str(data["cache_dir"]) if data.get("cache_dir") else None,
            diffusion_samples=int(data.get("diffusion_samples", data.get("n_samples", 5))),
            recycling_steps=int(data.get("recycling_steps", 3)),
            sampling_steps=int(data.get("sampling_steps", 200)),
            max_parallel_samples=int(data.get("max_parallel_samples", 1)),
            output_format=str(data.get("output_format", "pdb")),
            accelerator=str(data.get("accelerator", "gpu")),
            devices=int(data.get("devices", 1)),
            num_workers=int(data.get("num_workers", 0)),
            preprocessing_threads=int(data.get("preprocessing_threads", 1)),
            seed=int(data.get("seed", 0)),
            use_msa_server=bool(data.get("use_msa_server", False)),
            fp32=bool(data.get("fp32", False)),
            extra_args=tuple(str(x) for x in data.get("extra_args", ())),
        )


@dataclass(frozen=True)
class BoltzModelRef:
    """One predicted Boltz-2 structure and its confidence JSON."""

    design_id: str
    sample: int
    confidence_path: str
    structure_path: str | None = None
    mode: str = ""

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.design_id, self.mode, self.sample)


def build_boltz_input(
    sequence: str,
    ligands: Sequence[LigandSpec],
    protein_chain: str = "A",
    msa_path: str | Path | None = None,
) -> dict[str, Any]:
    """Construct a Boltz-2 v1 YAML/JSON input document for enzyme + ligands."""
    protein_entry: dict[str, Any] = {"id": protein_chain, "sequence": sequence}
    if msa_path is not None:
        protein_entry["msa"] = str(Path(msa_path).expanduser().resolve())
    sequences: list[dict[str, Any]] = [{"protein": protein_entry}]
    for lig in ligands:
        sequences.append(lig.to_boltz_entry())
    return {"version": 1, "sequences": sequences}


def write_boltz_input(document: dict[str, Any], path: str | Path) -> Path:
    """Write a Boltz-2 input YAML file atomically (JSON is valid YAML 1.2)."""
    dest = Path(path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n")
    tmp.replace(dest)
    return dest


def build_boltz_command(
    input_yaml: str | Path,
    out_dir: str | Path,
    config: BoltzConfig,
) -> list[str]:
    """Build the ``boltz predict`` command line."""
    cmd = [
        config.executable,
        "predict",
        str(Path(input_yaml).expanduser()),
        "--out_dir",
        str(Path(out_dir).expanduser()),
        "--diffusion_samples",
        str(config.diffusion_samples),
        "--max_parallel_samples",
        str(config.max_parallel_samples),
        "--output_format",
        config.output_format,
        "--accelerator",
        config.accelerator,
        "--devices",
        str(config.devices),
        "--num_workers",
        str(config.num_workers),
        "--preprocessing-threads",
        str(config.preprocessing_threads),
        "--seed",
        str(config.seed),
        "--recycling_steps",
        str(config.recycling_steps),
        "--sampling_steps",
        str(config.sampling_steps),
    ]
    if config.cache_dir:
        cmd.extend(["--cache", str(Path(config.cache_dir).expanduser())])
    if config.use_msa_server:
        cmd.append("--use_msa_server")
    cmd.extend(config.extra_args)
    return cmd


def run_boltz_predict(
    input_yaml: str | Path,
    out_dir: str | Path,
    config: BoltzConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Invoke ``boltz predict`` for one input YAML (or return the command if ``dry_run``)."""
    out_path = Path(out_dir).expanduser()
    cmd = build_boltz_command(input_yaml, out_path, config)
    if dry_run:
        return {"status": "dry_run", "command": cmd, "out_dir": str(out_path)}
    out_path.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "command": cmd,
        "out_dir": str(out_path),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _structure_beside(confidence_path: Path) -> str | None:
    stem = confidence_path.name[len("confidence_") : -len(".json")]
    for ext in (".pdb", ".cif", ".mmcif"):
        cand = confidence_path.parent / f"{stem}{ext}"
        if cand.is_file():
            return str(cand)
    return None


def discover_boltz_models(
    root: str | Path,
    known_ids: set[str] | None = None,
    modes: tuple[str, ...] = (),
) -> dict[tuple[str, str], list[BoltzModelRef]]:
    """Discover predicted Boltz-2 models under ``root`` without double counting.

    Recovers ``(design_id, sample)`` from ``confidence_<id>_model_<k>.json``
    across both stock ``boltz_results_<id>/predictions/<id>/`` and flat/seed
    directory layouts.
    """
    root_p = Path(root).expanduser()
    if not root_p.exists():
        return {}
    seen: set[tuple[str, str, int]] = set()
    groups: dict[tuple[str, str], list[BoltzModelRef]] = {}

    for dirpath, _dirnames, filenames in os.walk(root_p, followlinks=True):
        rel_parts = Path(dirpath).relative_to(root_p).parts if Path(dirpath) != root_p else ()
        mode = next((m for m in modes if m in rel_parts), "")
        for fn in sorted(filenames):
            match = CONFIDENCE_RE.match(fn)
            if not match:
                continue
            did, sample = match.group("id"), int(match.group("sample"))
            if known_ids is not None and did not in known_ids:
                continue
            key = (did, mode, sample)
            if key in seen:
                continue
            seen.add(key)
            cpath = Path(dirpath) / fn
            ref = BoltzModelRef(
                design_id=did,
                sample=sample,
                confidence_path=str(cpath),
                structure_path=_structure_beside(cpath),
                mode=mode,
            )
            groups.setdefault((did, mode), []).append(ref)

    for refs in groups.values():
        refs.sort(key=lambda r: r.sample)
    return groups


def read_confidence(
    confidence_path: str | Path,
    chain_pairs: dict[str, tuple[str, str]] | None = None,
    scalars: tuple[str, ...] = DEFAULT_CONFIDENCE_SCALARS,
) -> dict[str, Any]:
    """Parse Boltz-2 confidence JSON, including asymmetric ``pair_chains_iptm``."""
    try:
        data = json.loads(Path(confidence_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"confidence_error": f"{type(exc).__name__}: {exc}"}

    row: dict[str, Any] = {k: data.get(k) for k in scalars}
    pc = data.get("pair_chains_iptm") or {}
    for name, (a, b) in (chain_pairs or {}).items():
        vals = [
            pc[x][y]
            for x, y in ((a, b), (b, a))
            if isinstance(pc.get(x), dict) and pc[x].get(y) is not None
        ]
        row[name] = round(min(vals), 6) if vals else None
        if len(vals) == 2:
            row[f"{name}_asym"] = round(abs(vals[0] - vals[1]), 6)
    return row
