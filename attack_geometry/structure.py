"""Residue- and atom-level PDB and mmCIF parsing (stdlib + numpy).

Preserves 1-based residue numbering for mutagenesis while also retaining per-atom
names, elements, and coordinates for catalytic attack-geometry evaluation and
motif alignment. Supports PDB and mmCIF (``.cif``, ``.cif.gz``, ``.cif.zst``)
without requiring Biopython.
"""
from __future__ import annotations

import gzip
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import AtomSelector

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common protonation / alternate residue names
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "CYM": "C",
    "ASH": "D", "GLH": "E", "LYN": "K", "MSE": "M",
}

WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "TIP", "TIP3", "SOL"})


@dataclass(frozen=True)
class AtomRecord:
    """One heavy atom in a parsed structure."""

    record_type: str   # "ATOM" or "HETATM"
    chain: str
    resnum: int
    icode: str
    resname: str
    name: str
    element: str
    xyz: np.ndarray    # (3,)
    bfactor: float = 0.0
    source_label: str = ""


@dataclass(frozen=True)
class Residue:
    """One polymer residue with all heavy atoms it owns."""

    chain: str
    resnum: int
    resname: str
    aa1: str
    xyz: np.ndarray                      # (n_atoms, 3)
    atom_names: tuple[str, ...] = ()
    icode: str = ""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resnum, self.icode)


@dataclass(frozen=True)
class Component:
    """One non-polymer entity: substrate, intermediate, cofactor, or metal ion.

    ``id`` is the chain id when that chain holds a single non-polymer residue,
    or ``chain:RESNAME<resnum>`` when a chain holds several.
    """

    id: str
    chain: str
    resname: str
    resnum: int
    xyz: np.ndarray                      # (n_atoms, 3)
    elements: tuple[str, ...] = ()
    atom_names: tuple[str, ...] = ()

    @property
    def n_atoms(self) -> int:
        return int(self.xyz.shape[0])


@dataclass
class Structure:
    """Parsed polymer residues, non-polymer components, and flat heavy-atom list."""

    path: str
    residues: list[Residue] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)
    atoms: list[AtomRecord] = field(default_factory=list)

    def component(self, cid: str) -> Component | None:
        for c in self.components:
            if c.id == cid or c.chain == cid:
                return c
        return None

    def protein_chains(self) -> tuple[str, ...]:
        seen: list[str] = []
        for r in self.residues:
            if r.chain not in seen:
                seen.append(r.chain)
        return tuple(seen)

    def sequence(self, chain: str | None = None, strict_numbering: bool = True) -> str:
        """Return 1-letter protein sequence for ``chain``.

        When ``strict_numbering=True`` (default), asserts gap-free 1-based
        numbering with no insertion codes so positional mutations are exact.
        """
        chain = chain or (self.protein_chains() or ("",))[0]
        rs = [r for r in self.residues if r.chain == chain]
        if not rs:
            raise ValueError(f"{self.path}: no protein residues in chain {chain!r}")
        if strict_numbering:
            icoded = [r for r in rs if r.icode]
            if icoded:
                raise ValueError(
                    f"{self.path}: chain {chain!r} has insertion codes "
                    f"(e.g. {icoded[0].resnum}{icoded[0].icode}); residue number is not a "
                    "sequence index, so mutations cannot be applied positionally"
                )
            nums = [r.resnum for r in rs]
            if nums[0] != 1:
                raise ValueError(
                    f"{self.path}: chain {chain!r} numbering starts at {nums[0]}, expected 1"
                )
            if nums != list(range(1, nums[-1] + 1)):
                gaps = [b for a, b in zip(nums, nums[1:]) if b != a + 1]
                raise ValueError(
                    f"{self.path}: chain {chain!r} has a gap in residue numbering before "
                    f"{gaps[:5]}"
                )
        unknown = [r.resnum for r in rs if r.aa1 == "X"]
        if unknown:
            raise ValueError(
                f"{self.path}: chain {chain!r} has non-standard residues at {unknown[:5]}"
            )
        return "".join(r.aa1 for r in rs)

    def atom_map(self, chain: str | None = None) -> dict[tuple[int, str], np.ndarray]:
        """Map ``(1-based sequential index, atom_name) -> xyz`` for a protein chain."""
        chain = chain or (self.protein_chains() or ("",))[0]
        rs = [r for r in self.residues if r.chain == chain]
        key_to_pos = {r.key: idx for idx, r in enumerate(rs, start=1)}
        out: dict[tuple[int, str], np.ndarray] = {}
        for a in self.atoms:
            if a.record_type == "ATOM" and a.chain == chain:
                pos = key_to_pos.get((a.chain, a.resnum, a.icode))
                if pos is not None:
                    out[(pos, a.name)] = a.xyz
        return out


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    if path.suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:
            raise ImportError(
                "Reading .zst structures requires the `zstandard` package."
            ) from exc
        with path.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as stream:
            return stream.read().decode("utf-8")
    return path.read_text()


def _pdb_element(line: str, name: str) -> str:
    return (line[76:78].strip() or name.strip()[:1]).upper()


def _parse_pdb_atoms(text: str, keep_water: bool) -> list[AtomRecord]:
    atoms: list[AtomRecord] = []
    seen_alt: set[tuple] = set()
    for line in text.splitlines():
        rec = line[:6]
        if rec == "ENDMDL":
            break
        if rec not in ("ATOM  ", "HETATM"):
            continue
        name = line[12:16].strip()
        elem = _pdb_element(line, name)
        if elem in ("H", "D"):
            continue
        altloc = line[16]
        chain = line[21]
        resname = line[17:20].strip()
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26].strip()
        atom_key = (chain, resnum, icode, resname, name)
        if altloc not in (" ", ""):
            if atom_key in seen_alt:
                continue
            seen_alt.add(atom_key)
        is_polymer = rec == "ATOM  " or resname == "MSE"
        if not is_polymer and not keep_water and resname in WATER_RESNAMES:
            continue
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        )
        try:
            bfactor = float(line[60:66]) if len(line) >= 66 and line[60:66].strip() else 0.0
        except ValueError:
            bfactor = 0.0
        atoms.append(
            AtomRecord(
                record_type="ATOM" if is_polymer else "HETATM",
                chain=chain,
                resnum=resnum,
                icode=icode,
                resname=resname,
                name=name,
                element=elem,
                xyz=xyz,
                bfactor=bfactor,
            )
        )
    return atoms


def _parse_mmcif_atoms(text: str, keep_water: bool) -> list[AtomRecord]:
    """Lightweight pure-Python mmCIF ``_atom_site`` loop parser (model 1)."""
    lines = text.splitlines()
    columns: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    reading_cols = False
    reading_rows = False

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#"):
            continue
        if raw == "loop_":
            in_loop = True
            reading_cols = True
            reading_rows = False
            columns = []
            continue
        if in_loop and reading_cols and raw.startswith("_atom_site."):
            columns.append(raw.split(".", 1)[1])
            continue
        if in_loop and reading_cols and raw.startswith("_"):
            if columns:
                # Different category started
                break
            in_loop = False
            reading_cols = False
            continue
        if in_loop and columns:
            reading_cols = False
            reading_rows = True
            if raw.startswith("_") or raw == "loop_" or raw.startswith("data_"):
                break
            tokens = shlex.split(raw)
            while len(tokens) < len(columns) and i < n_lines:
                nxt = lines[i].strip()
                i += 1
                if nxt and not nxt.startswith("#"):
                    tokens.extend(shlex.split(nxt))
            if len(tokens) >= len(columns):
                rows.append(tokens[: len(columns)])
        elif reading_rows:
            break

    if not columns or not rows:
        raise ValueError("No _atom_site loop found in mmCIF file")

    idx = {col: j for j, col in enumerate(columns)}

    def _get(row: list[str], *candidates: str, default: str = "") -> str:
        for c in candidates:
            j = idx.get(c)
            if j is not None:
                val = row[j]
                if val not in (".", "?"):
                    return val
        return default

    atoms: list[AtomRecord] = []
    seen_alt: set[tuple] = set()
    first_model: str | None = None
    for row in rows:
        model_num = _get(row, "pdbx_PDB_model_num", default="1")
        if first_model is None:
            first_model = model_num
        elif model_num != first_model:
            continue

        group_pdb = _get(row, "group_PDB", default="ATOM").upper()
        name = _get(row, "auth_atom_id", "label_atom_id")
        elem = (_get(row, "type_symbol") or name[:1]).upper()
        if elem in ("H", "D"):
            continue
        altloc = _get(row, "label_alt_id", default="")
        chain = _get(row, "auth_asym_id", "label_asym_id", default="A")
        resname = _get(row, "auth_comp_id", "label_comp_id")
        seq_str = _get(row, "auth_seq_id", "label_seq_id", default="1")
        try:
            resnum = int(seq_str)
        except ValueError:
            resnum = 1
        icode = _get(row, "pdbx_PDB_ins_code", default="")
        atom_key = (chain, resnum, icode, resname, name)
        if altloc:
            if atom_key in seen_alt:
                continue
            seen_alt.add(atom_key)

        is_polymer = group_pdb == "ATOM" or resname == "MSE"
        if not is_polymer and not keep_water and resname in WATER_RESNAMES:
            continue

        xyz = np.array(
            [
                float(_get(row, "Cartn_x")),
                float(_get(row, "Cartn_y")),
                float(_get(row, "Cartn_z")),
            ],
            dtype=float,
        )
        bf_str = _get(row, "B_iso_or_equiv", default="0.0")
        try:
            bfactor = float(bf_str)
        except ValueError:
            bfactor = 0.0
        source_label = _get(row, "annotation_src-coordinate-conditioned-region_1_atom", default="")
        atoms.append(
            AtomRecord(
                record_type="ATOM" if is_polymer else "HETATM",
                chain=chain,
                resnum=resnum,
                icode=icode,
                resname=resname,
                name=name,
                element=elem,
                xyz=xyz,
                bfactor=bfactor,
                source_label=source_label,
            )
        )
    return atoms


def _is_mmcif_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in (".cif", ".mmcif", ".cif.gz", ".mmcif.gz", ".cif.zst"))


def load_structure(path: str | Path, keep_water: bool = False) -> Structure:
    """Parse a PDB or mmCIF file into polymer residues, components, and atoms."""
    p = Path(path).expanduser()
    text = _read_text(p)
    atoms = _parse_mmcif_atoms(text, keep_water) if _is_mmcif_path(p) else _parse_pdb_atoms(text, keep_water)

    residues: dict[tuple[str, int, str], dict] = {}
    het: dict[tuple[str, str, int], dict] = {}
    for a in atoms:
        if a.record_type == "ATOM":
            k = (a.chain, a.resnum, a.icode)
            d = residues.setdefault(
                k,
                {"resname": a.resname, "aa1": THREE_TO_ONE.get(a.resname, "X"), "xyz": [], "names": []},
            )
            d["xyz"].append(a.xyz)
            d["names"].append(a.name)
        else:
            k2 = (a.chain, a.resname, a.resnum)
            d2 = het.setdefault(k2, {"xyz": [], "elements": [], "names": []})
            d2["xyz"].append(a.xyz)
            d2["elements"].append(a.element)
            d2["names"].append(a.name)

    res_list = [
        Residue(
            chain=c,
            resnum=n,
            resname=d["resname"],
            aa1=d["aa1"],
            xyz=np.asarray(d["xyz"], dtype=float),
            atom_names=tuple(d["names"]),
            icode=ic,
        )
        for (c, n, ic), d in sorted(residues.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    ]

    per_chain: dict[str, int] = {}
    for (c, _rn, _num) in het:
        per_chain[c] = per_chain.get(c, 0) + 1
    comps = []
    for (c, resname, num), d in sorted(het.items()):
        cid = c if per_chain[c] == 1 else f"{c}:{resname}{num}"
        comps.append(
            Component(
                id=cid,
                chain=c,
                resname=resname,
                resnum=num,
                xyz=np.asarray(d["xyz"], dtype=float),
                elements=tuple(d["elements"]),
                atom_names=tuple(d["names"]),
            )
        )

    return Structure(path=str(p), residues=res_list, components=comps, atoms=atoms)


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum heavy-atom to heavy-atom distance between two point sets (Å)."""
    if a.size == 0 or b.size == 0:
        return float("inf")
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min()))


def select_atoms(structure: Structure, selector: AtomSelector) -> list[AtomRecord]:
    """Select heavy atoms matching ``selector``, enforcing optional count checks."""
    names_set = set(selector.atom_names)
    if selector.atom is not None:
        names_set.add(selector.atom)
    elements_set = {e.upper() for e in selector.elements}

    matched: list[AtomRecord] = []
    seen_names: set[tuple[str, int, str, str]] = set()
    for a in structure.atoms:
        if a.chain != selector.chain:
            continue
        if selector.record_type and a.record_type != selector.record_type:
            continue
        if selector.resnum is not None and a.resnum != selector.resnum:
            continue
        if names_set and a.name not in names_set:
            continue
        if elements_set and a.element.upper() not in elements_set:
            continue
        key = (a.chain, a.resnum, a.icode, a.name)
        if key in seen_names:
            raise ValueError(f"{structure.path}: duplicate atom {key}")
        seen_names.add(key)
        matched.append(a)

    if not matched:
        raise ValueError(f"{structure.path}: selector matched 0 atoms: {selector.to_dict()}")
    if names_set:
        found_names = {a.name for a in matched}
        if found_names != names_set:
            missing = sorted(names_set - found_names)
            raise ValueError(
                f"{structure.path}: selector missing atom(s) {missing} for {selector.to_dict()}"
            )
    if selector.n_atoms is not None and len(matched) != selector.n_atoms:
        raise ValueError(
            f"{structure.path}: selector matched {len(matched)} atoms, expected {selector.n_atoms} "
            f"for {selector.to_dict()}"
        )
    # Deterministic ordering: lexical by (chain, resnum, name)
    matched.sort(key=lambda a: (a.chain, a.resnum, a.name))
    return matched
