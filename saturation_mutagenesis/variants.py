"""Build single-site and multi-site saturation mutagenesis variant libraries.

Catalytic positions believed essential are retained (and optionally annotated)
rather than silently masked out, serving as internal negative controls.
"""
from __future__ import annotations

import csv
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

AA20 = "ACDEFGHIKLMNPQRSTVWY"
WT_MUTATION = "WT"
_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    kind: str                            # "wt" | "single" | "multi"
    mutation: str                        # "WT", "S190T", or "D143E_S190T"
    resnum: int | None
    wt_aa: str
    mut_aa: str
    seq: str
    n_mut: int = 1
    annotation: str = ""
    needs: dict[str, bool] = field(default_factory=dict)

    @property
    def needed_modes(self) -> tuple[str, ...]:
        return tuple(m for m, need in self.needs.items() if need)


def mutate(seq: str, resnum: int, wt_aa: str, mut_aa: str) -> str:
    """Apply a single 1-based substitution, verifying the wild-type residue."""
    i = resnum - 1
    if not 0 <= i < len(seq):
        raise ValueError(f"Residue {resnum} is outside sequence of length {len(seq)}")
    if seq[i] != wt_aa:
        raise ValueError(
            f"Residue {resnum} is {seq[i]!r} in sequence, expected {wt_aa!r} "
            "(numbering mismatch between structure and sequence)"
        )
    if mut_aa == wt_aa:
        raise ValueError(f"{wt_aa}{resnum}{mut_aa} is not a substitution")
    return seq[:i] + mut_aa + seq[i + 1 :]


def mutate_many(seq: str, substitutions: Sequence[tuple[int, str, str]]) -> str:
    """Apply multiple non-overlapping 1-based substitutions ``(resnum, wt_aa, mut_aa)``."""
    seen_pos: set[int] = set()
    chars = list(seq)
    for resnum, wt_aa, mut_aa in substitutions:
        if resnum in seen_pos:
            raise ValueError(f"Duplicate mutation at position {resnum}")
        seen_pos.add(resnum)
        i = resnum - 1
        if not 0 <= i < len(seq):
            raise ValueError(f"Residue {resnum} is outside sequence of length {len(seq)}")
        if chars[i] != wt_aa:
            raise ValueError(
                f"Residue {resnum} is {chars[i]!r} in sequence, expected {wt_aa!r}"
            )
        if mut_aa == wt_aa:
            raise ValueError(f"{wt_aa}{resnum}{mut_aa} is not a substitution")
        chars[i] = mut_aa
    return "".join(chars)


def parse_mutation_string(mut_str: str) -> list[tuple[int, str, str]]:
    """Parse ``'S190T'`` or ``'D143E+S190T'`` / ``'D143E_S190T'`` into tuples."""
    tokens = [t.strip() for t in re.split(r"[+_,/]", mut_str.strip()) if t.strip()]
    out: list[tuple[int, str, str]] = []
    for tok in tokens:
        m = _MUT_RE.match(tok.upper())
        if not m:
            raise ValueError(f"Invalid mutation token: {tok!r} (expected e.g. 'S190T')")
        out.append((int(m.group(2)), m.group(1), m.group(3)))
    out.sort(key=lambda x: x[0])
    return out


def _normalize_positions(seq: str, positions: Iterable) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for p in positions:
        if hasattr(p, "resnum"):
            pairs.append((int(p.resnum), str(p.wt_aa)))
        elif isinstance(p, int):
            if not 1 <= p <= len(seq):
                raise ValueError(f"Position {p} outside sequence length {len(seq)}")
            pairs.append((p, seq[p - 1]))
        else:
            rn, wt = p
            pairs.append((int(rn), str(wt)))
    return pairs


def _check_unique(variants: list[Variant]) -> None:
    ids = [v.id for v in variants]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate variant ids: {dupes[:5]}")
    seqs: dict[str, str] = {}
    for v in variants:
        if v.seq in seqs:
            raise ValueError(f"Variants {seqs[v.seq]} and {v.id} have identical sequences")
        seqs[v.seq] = v.id


def build_variants(
    seq: str,
    positions: Iterable,
    label: str = "",
    alphabet: str = AA20,
    modes: tuple[str, ...] = (),
    existing: set[tuple[str, str]] | None = None,
    dedup: bool = False,
    annotate: Callable[[int, str], str] | None = None,
    include_wt: bool = True,
) -> list[Variant]:
    """Generate every single-site substitution across ``positions``."""
    existing = existing or set()
    pairs = _normalize_positions(seq, positions)

    def _needs(vid: str) -> dict[str, bool]:
        return {m: not (dedup and (vid, m) in existing) for m in modes}

    out: list[Variant] = []
    if include_wt:
        wid = f"{label}_{WT_MUTATION}" if label else WT_MUTATION
        out.append(
            Variant(
                id=wid,
                label=label,
                kind="wt",
                mutation=WT_MUTATION,
                resnum=None,
                wt_aa="",
                mut_aa="",
                seq=seq,
                n_mut=0,
                annotation="",
                needs=_needs(wid),
            )
        )

    for resnum, wt_aa in pairs:
        ann = annotate(resnum, wt_aa) if annotate else ""
        for mut_aa in alphabet:
            if mut_aa == wt_aa:
                continue
            mutation = f"{wt_aa}{resnum}{mut_aa}"
            vid = f"{label}_{mutation}" if label else mutation
            out.append(
                Variant(
                    id=vid,
                    label=label,
                    kind="single",
                    mutation=mutation,
                    resnum=resnum,
                    wt_aa=wt_aa,
                    mut_aa=mut_aa,
                    seq=mutate(seq, resnum, wt_aa, mut_aa),
                    n_mut=1,
                    annotation=ann,
                    needs=_needs(vid),
                )
            )

    _check_unique(out)
    return out


def build_multisite_variants(
    seq: str,
    positions: Iterable | None = None,
    *,
    label: str = "",
    alphabet: str = AA20,
    min_mutations: int = 1,
    max_mutations: int = 2,
    explicit_mutations: Sequence[str] | None = None,
    modes: tuple[str, ...] = (),
    annotate: Callable[[int, str], str] | None = None,
    include_wt: bool = True,
    max_variants: int | None = None,
) -> list[Variant]:
    """Generate single- and multi-site variants (combinatorial or explicit list).

    Args:
        seq: Wild-type 1-based protein sequence.
        positions: Pocket ``Position`` objects, ``(resnum, wt_aa)`` pairs, or
            1-based residue numbers for combinatorial mutagenesis.
        explicit_mutations: Optional explicit mutation strings such as
            ``["D143E", "D143E+S190T"]``. If provided, generates exactly those
            variants (plus optional WT) instead of enumerating combinations.
        min_mutations / max_mutations: Mutation order range for combinatorial mode.
        max_variants: Optional safety cap on combinatorial library size.
    """
    def _needs(vid: str) -> dict[str, bool]:
        return {m: True for m in modes}

    out: list[Variant] = []
    if include_wt:
        wid = f"{label}_{WT_MUTATION}" if label else WT_MUTATION
        out.append(
            Variant(
                id=wid,
                label=label,
                kind="wt",
                mutation=WT_MUTATION,
                resnum=None,
                wt_aa="",
                mut_aa="",
                seq=seq,
                n_mut=0,
                annotation="",
                needs=_needs(wid),
            )
        )

    if explicit_mutations is not None:
        for mut_str in explicit_mutations:
            subs = parse_mutation_string(mut_str)
            mut_label = "_".join(f"{wt}{rn}{ma}" for rn, wt, ma in subs)
            vid = f"{label}_{mut_label}" if label else mut_label
            anns = [annotate(rn, wt) for rn, wt, _ in subs if annotate]
            ann = ";".join(a for a in anns if a)
            out.append(
                Variant(
                    id=vid,
                    label=label,
                    kind="single" if len(subs) == 1 else "multi",
                    mutation=mut_label,
                    resnum=subs[0][0] if len(subs) == 1 else None,
                    wt_aa=subs[0][1] if len(subs) == 1 else "",
                    mut_aa=subs[0][2] if len(subs) == 1 else "",
                    seq=mutate_many(seq, subs),
                    n_mut=len(subs),
                    annotation=ann,
                    needs=_needs(vid),
                )
            )
        _check_unique(out)
        return out

    if positions is None:
        raise ValueError("Provide either 'positions' or 'explicit_mutations'.")
    pairs = _normalize_positions(seq, positions)
    if min_mutations < 1 or max_mutations < min_mutations:
        raise ValueError("Require 1 <= min_mutations <= max_mutations")

    for k in range(min_mutations, min(max_mutations, len(pairs)) + 1):
        for site_combo in itertools.combinations(pairs, k):
            aa_choices = [[aa for aa in alphabet if aa != wt] for _rn, wt in site_combo]
            for mut_aas in itertools.product(*aa_choices):
                subs = [(rn, wt, ma) for (rn, wt), ma in zip(site_combo, mut_aas)]
                mut_label = "_".join(f"{wt}{rn}{ma}" for rn, wt, ma in subs)
                vid = f"{label}_{mut_label}" if label else mut_label
                anns = [annotate(rn, wt) for rn, wt, _ in subs if annotate]
                ann = ";".join(a for a in anns if a)
                out.append(
                    Variant(
                        id=vid,
                        label=label,
                        kind="single" if k == 1 else "multi",
                        mutation=mut_label,
                        resnum=subs[0][0] if k == 1 else None,
                        wt_aa=subs[0][1] if k == 1 else "",
                        mut_aa=subs[0][2] if k == 1 else "",
                        seq=mutate_many(seq, subs),
                        n_mut=k,
                        annotation=ann,
                        needs=_needs(vid),
                    )
                )
                if max_variants is not None and len(out) > max_variants:
                    raise ValueError(
                        f"Combinatorial library exceeded max_variants={max_variants}; "
                        "narrow positions, alphabet, or max_mutations."
                    )

    _check_unique(out)
    return out


VARIANT_FIELDS = (
    "id", "label", "kind", "mutation", "resnum", "wt_aa", "mut_aa",
    "annotation", "n_mut", "seq",
)


def write_variants_tsv(
    variants: list[Variant],
    path: str | Path,
    modes: tuple[str, ...] = (),
) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(VARIANT_FIELDS) + [f"need_{m}" for m in modes]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for v in variants:
            row = {
                "id": v.id,
                "label": v.label,
                "kind": v.kind,
                "mutation": v.mutation,
                "resnum": v.resnum if v.resnum is not None else "",
                "wt_aa": v.wt_aa,
                "mut_aa": v.mut_aa,
                "annotation": v.annotation,
                "n_mut": v.n_mut,
                "seq": v.seq,
            }
            for m in modes:
                row[f"need_{m}"] = int(bool(v.needs.get(m, True)))
            w.writerow(row)
    return p


def read_variants_tsv(path: str | Path) -> list[Variant]:
    """Read a variant library TSV previously written by ``write_variants_tsv``."""
    p = Path(path).expanduser()
    out: list[Variant] = []
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            resnum_raw = row.get("resnum", "").strip()
            out.append(
                Variant(
                    id=row["id"],
                    label=row.get("label", ""),
                    kind=row.get("kind", "single"),
                    mutation=row["mutation"],
                    resnum=int(resnum_raw) if resnum_raw else None,
                    wt_aa=row.get("wt_aa", ""),
                    mut_aa=row.get("mut_aa", ""),
                    seq=row["seq"],
                    n_mut=int(row.get("n_mut", 0 if row["mutation"] == WT_MUTATION else 1)),
                    annotation=row.get("annotation", ""),
                )
            )
    return out


def write_variants_fasta(
    variants: list[Variant],
    path: str | Path,
    needed_only: bool = True,
) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for v in variants:
        if needed_only and v.needs and not v.needed_modes:
            continue
        lines.append(f">{v.id}")
        lines.extend(v.seq[i : i + 60] for i in range(0, len(v.seq), 60))
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    return p
