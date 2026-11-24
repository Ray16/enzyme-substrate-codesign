"""Resumable batched MSA retrieval with sequence-SHA256 content-addressed caching.

Extracted cleanly from ``structural_modeling_agent/msa.py``. Uses the public
ColabFold MMseqs2 API (configurable) in batches of up to 101 unique sequences,
validating archive members and exact query-sequence agreement before caching.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import re
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_MSA_SERVER = "https://api.colabfold.com"
ALIGNMENT_FILES = ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    os.replace(tmp, path)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(data, indent=2) + "\n")


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def valid_msa_csv(path: str | Path, sequence: str) -> bool:
    """Verify a cached Boltz-2 MSA CSV matches ``sequence`` and its SHA-256 metadata."""
    try:
        p = Path(path)
        meta = json.loads(p.with_suffix(".json").read_text())
        if meta.get("alignment_policy") != "full_server_alignment_v1":
            return False
        if (
            meta.get("csv_sha256") != sha256_bytes(p.read_bytes())
            or meta.get("sequence_sha256") != sha256_bytes(sequence.encode("utf-8"))
        ):
            return False
        with p.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        return (
            len(rows) >= 1
            and rows[0].get("sequence") == sequence
            and all(row.get("key") == "-1" and bool(row.get("sequence")) for row in rows)
        )
    except (OSError, ValueError):
        return False


def alignment_csvs(archive: str | Path, sequences: Sequence[str]) -> list[str]:
    """Parse named A3M members from a ColabFold tar.gz archive into Boltz-2 CSV strings."""
    results: dict[int, list[str]] = {101 + i: [] for i in range(len(sequences))}
    with tarfile.open(archive, "r:gz") as tar:
        for filename in ALIGNMENT_FILES:
            member = tar.getmember(filename)
            if not member.isfile() or member.size > 512 * 1024 * 1024:
                raise ValueError(f"Invalid alignment archive member: {filename}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Unable to read archive member: {filename}")
            text = stream.read().decode("utf-8")
            current: int | None = None
            new_block = True
            for raw in text.splitlines():
                if "\x00" in raw:
                    new_block = True
                line = raw.replace("\x00", "").strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if new_block:
                        current = int(line[1:])
                        new_block = False
                    continue
                if current not in results:
                    raise ValueError(f"Unexpected alignment query identifier: {current}")
                results[current].append(line)

    outputs: list[str] = []
    for index, sequence in enumerate(sequences, start=101):
        rows = results[index]
        if not rows or rows[0] != sequence:
            raise ValueError("Alignment query does not match its exact source sequence")
        outputs.append(
            "key,sequence\n" + "".join(f"-1,{row}\n" for row in dict.fromkeys(rows))
        )
    return outputs


def _request_json(session: Any, method: str, url: str, *, data: dict | None = None) -> dict:
    import requests

    for attempt in range(4):
        try:
            response = session.request(method, url, data=data, timeout=(30, 180))
            if response.status_code in {429, 502, 503, 504} and method == "GET":
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except requests.ConnectTimeout:
            if attempt == 3:
                raise
        except (requests.ReadTimeout, requests.ConnectionError, requests.HTTPError):
            if method != "GET" or attempt == 3:
                raise
        time.sleep(min(10 * (2**attempt), 40))
    raise RuntimeError(f"Exhausted retries for {method} {url}")


def fetch_msa_batch(
    sequences: Sequence[str],
    cache_dir: str | Path,
    *,
    session: Any = None,
    timeout: int = 1800,
    server: str = DEFAULT_MSA_SERVER,
) -> list[Path]:
    """Fetch and cache Boltz-2 MSA CSVs for 1..101 unique sequences."""
    import requests

    if not 1 <= len(sequences) <= 101:
        raise ValueError("A batch must contain 1..101 distinct sequences")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Deduplicate sequences before submission")

    cache = Path(cache_dir).expanduser()
    identity = sha256_bytes(
        json.dumps({"sequences": list(sequences), "server": server, "mode": "env"}).encode()
    )
    folder = cache / "batches" / identity
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "request.json"
    archive = folder / "out.tar.gz"
    sess = session or requests.Session()

    with _file_lock(folder / "lock"):
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"identity": identity, "status": "new"}
        )
        if not archive.exists():
            if state.get("status") in {"submission_unknown", "submitting"} and not state.get("ticket"):
                raise RuntimeError(
                    "Prior submission response was ambiguous; inspect before resubmitting"
                )
            if not state.get("ticket"):
                query = "".join(f">{101 + i}\n{seq}\n" for i, seq in enumerate(sequences))
                state.update(
                    status="submitting",
                    requested_at=time.time(),
                    sequence_hashes=[sha256_bytes(s.encode()) for s in sequences],
                )
                _save_json(state_path, state)
                try:
                    reply = _request_json(
                        sess, "POST", f"{server}/ticket/msa", data={"q": query, "mode": "env"}
                    )
                except requests.ConnectTimeout:
                    state.update(status="connection_failed")
                    _save_json(state_path, state)
                    raise
                except Exception:
                    state.update(status="submission_unknown")
                    _save_json(state_path, state)
                    raise
                if not reply.get("id"):
                    state.update(status="server_rejected", response=reply)
                    _save_json(state_path, state)
                    raise RuntimeError(
                        f"Alignment server did not accept batch: {reply.get('status')}"
                    )
                state.update(ticket=reply["id"], status=reply.get("status", "PENDING"))
                _save_json(state_path, state)

            deadline = time.monotonic() + timeout
            while state["status"] != "COMPLETE":
                if state["status"] in {"ERROR", "MAINTENANCE"}:
                    raise RuntimeError(f"Alignment server returned {state['status']}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Alignment batch checkpointed while waiting for {state['ticket']}"
                    )
                reply = _request_json(sess, "GET", f"{server}/ticket/{state['ticket']}")
                state.update(status=reply["status"], checked_at=time.time())
                _save_json(state_path, state)
                if state["status"] != "COMPLETE":
                    time.sleep(10)

            resp = sess.get(f"{server}/result/download/{state['ticket']}", timeout=(30, 180))
            resp.raise_for_status()
            _atomic_write(archive, resp.content)

        outputs = alignment_csvs(archive, sequences)
        state.update(status="validated", archive_sha256=sha256_bytes(archive.read_bytes()))
        _save_json(state_path, state)

        paths: list[Path] = []
        for sequence, content in zip(sequences, outputs):
            seq_hash = sha256_bytes(sequence.encode())
            dest = cache / f"{seq_hash}.csv"
            _atomic_write(dest, content)
            _save_json(
                dest.with_suffix(".json"),
                {
                    "sequence_sha256": seq_hash,
                    "csv_sha256": sha256_bytes(content.encode()),
                    "server": server,
                    "batch": identity,
                    "ticket": state.get("ticket"),
                    "aligned_rows": content.count("\n") - 1,
                    "alignment_policy": "full_server_alignment_v1",
                },
            )
            paths.append(dest)
        return paths


def parse_fasta(fasta_path: str | Path) -> dict[str, str]:
    """Read a FASTA file into an ordered ``{id: sequence}`` mapping."""
    records: dict[str, str] = {}
    ident: str | None = None
    for line in Path(fasta_path).expanduser().read_text().splitlines():
        if line.startswith(">"):
            ident = line[1:].split()[0]
            records[ident] = ""
        elif line.strip():
            if ident is None:
                raise ValueError("FASTA sequence precedes its header")
            records[ident] += line.strip().upper()
    if not records or any(not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", s) for s in records.values()):
        raise ValueError("FASTA must contain non-empty standard amino acid sequences")
    return records


def prepare_msas_from_fasta(
    fasta_path: str | Path,
    out_dir: str | Path,
    cache_dir: str | Path,
    *,
    fetcher: Callable[[Sequence[str], Path], list[Path]] = fetch_msa_batch,
) -> list[dict[str, str]]:
    """Fetch missing MSAs for all sequences in ``fasta_path`` and write ``index.tsv``."""
    records = parse_fasta(fasta_path)
    cache = Path(cache_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    unique = list(dict.fromkeys(records.values()))
    missing = [s for s in unique if not valid_msa_csv(cache / f"{sha256_bytes(s.encode())}.csv", s)]
    for offset in range(0, len(missing), 101):
        fetcher(missing[offset : offset + 101], cache)

    rows: list[dict[str, str]] = []
    for ident, seq in records.items():
        checksum = sha256_bytes(seq.encode())
        csv_path = cache / f"{checksum}.csv"
        if not valid_msa_csv(csv_path, seq):
            raise ValueError(f"Missing or invalid cached MSA for {ident}")
        rows.append({"id": ident, "seqhash": checksum, "csv": str(csv_path)})

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "seqhash", "csv"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(out / "index.tsv", buf.getvalue())
    return rows
