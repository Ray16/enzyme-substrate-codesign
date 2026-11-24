"""Offline tests for standalone Boltz-2 co-folding (MSA caching, FP32 patch, QC, CLI)."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from boltz_cofold.cli import build_parser, main as cofold_cli_main
from boltz_cofold.msa import alignment_csvs, prepare_msas_from_fasta, sha256_bytes, valid_msa_csv
from boltz_cofold.predict import (
    batch_prepare_boltz_inputs,
    serial_sampler_source,
    verify_cofold_outputs,
)
from attack_geometry.config import LigandSpec

from ._synthetic import SEQUENCE, write_mock_boltz_prediction


def test_serial_sampler_source_patch():
    sample_src = "for batch in sample_ids.chunk(multiplicity % max_parallel_samples + 1):\n    pass\n"
    fixed, changed = serial_sampler_source(sample_src)
    assert changed is True
    assert "sample_ids.split(max_parallel_samples)" in fixed

    fixed2, changed2 = serial_sampler_source(fixed)
    assert changed2 is False


def test_msa_archive_parsing_and_cache(tmp_path: Path):
    archive = tmp_path / "out.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, body in (
            ("uniref.a3m", f">101\n{SEQUENCE}\n>hit1\nADGVS\n"),
            ("bfd.mgnify30.metaeuk30.smag30.a3m", f">101\n{SEQUENCE}\n>hit2\nAEGVT\n"),
        ):
            raw = body.encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))

    csvs = alignment_csvs(archive, [SEQUENCE])
    assert len(csvs) == 1
    assert f"-1,{SEQUENCE}" in csvs[0]
    assert "-1,ADGVS" in csvs[0]
    assert "-1,AEGVT" in csvs[0]

    # Exercise prepare_msas_from_fasta with a mock offline fetcher
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(f">enz1\n{SEQUENCE}\n")
    cache = tmp_path / "msa_cache"

    def _mock_fetch(seqs, cache_dir):
        out = []
        for s in seqs:
            h = sha256_bytes(s.encode())
            p = Path(cache_dir) / f"{h}.csv"
            content = f"key,sequence\n-1,{s}\n-1,ADGVS\n"
            p.write_text(content)
            p.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "sequence_sha256": h,
                        "csv_sha256": sha256_bytes(content.encode()),
                        "alignment_policy": "full_server_alignment_v1",
                    }
                )
            )
            out.append(p)
        return out

    rows = prepare_msas_from_fasta(fasta, tmp_path / "msa_out", cache, fetcher=_mock_fetch)
    assert len(rows) == 1
    assert valid_msa_csv(rows[0]["csv"], SEQUENCE) is True
    assert (tmp_path / "msa_out" / "index.tsv").is_file()


def test_cofold_prepare_verify_and_cli(tmp_path: Path):
    assert build_parser().prog == "boltz-cofold"
    fasta = tmp_path / "cand.fasta"
    fasta.write_text(f">cand1\n{SEQUENCE}\n")
    cfg = tmp_path / "cofold.json"
    cfg.write_text(json.dumps({"ligands": [{"id": "B", "ccd": "ATP"}, {"id": "D", "smiles": "O=C(O)CCCCC(=O)O"}]}))

    rc = cofold_cli_main(
        ["prepare-inputs", "--fasta", str(fasta), "--config", str(cfg), "--output-dir", str(tmp_path)]
    )
    assert rc == 0
    assert (tmp_path / "boltz_inputs" / "cand1.yaml").is_file()

    rc_dry = cofold_cli_main(
        [
            "predict",
            "--inputs-dir",
            str(tmp_path / "boltz_inputs"),
            "--results-dir",
            str(tmp_path / "boltz_results"),
            "--fp32",
            "--dry-run",
        ]
    )
    assert rc_dry == 0

    write_mock_boltz_prediction(tmp_path / "boltz_results", "cand1", SEQUENCE, offsets=[(-3.0, 0.0, 0.0)])
    summary = verify_cofold_outputs(
        tmp_path / "boltz_results",
        expected_sequences={"cand1": SEQUENCE},
        expected_samples=1,
        output_dir=tmp_path / "qc",
    )
    assert summary["n_verified"] == 1
    assert summary["n_complete"] == 1
    assert (tmp_path / "qc" / "cofold_verification.csv").is_file()
