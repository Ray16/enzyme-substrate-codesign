"""Configuration models and YAML/JSON loading for enzyme design workflows."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    p = Path(path).expanduser()
    text = p.read_text()
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files (`pip install pyyaml`), "
                "or provide a .json config file."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {p}")
    return data


@dataclass(frozen=True)
class AtomSelector:
    """Select one or more heavy atoms in a structure.

    At least one of ``atom``, ``atom_names``, or ``elements`` should be supplied
    when identifying reactive atoms. ``resnum`` (or ``res_id``) is optional and
    disambiguates polymer residues or multi-residue chains.
    """

    chain: str
    resnum: int | None = None
    atom: str | None = None
    atom_names: tuple[str, ...] = ()
    elements: tuple[str, ...] = ()
    n_atoms: int | None = None
    record_type: str | None = None  # "ATOM" | "HETATM" | None (any)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomSelector":
        if "chain" not in data:
            raise ValueError(f"AtomSelector requires 'chain': {data}")
        resnum = data.get("resnum", data.get("res_id"))
        atom = data.get("atom")
        atom_names = tuple(data.get("atom_names", data.get("atoms", ())))
        elements = tuple(str(e).upper() for e in data.get("elements", ()))
        if "element" in data and data["element"]:
            elements = tuple(dict.fromkeys((*elements, str(data["element"]).upper())))
        n_atoms = data.get("n_atoms", data.get("expected_count"))
        return cls(
            chain=str(data["chain"]),
            resnum=int(resnum) if resnum is not None else None,
            atom=str(atom) if atom is not None else None,
            atom_names=atom_names,
            elements=elements,
            n_atoms=int(n_atoms) if n_atoms is not None else None,
            record_type=str(data["record_type"]).upper() if data.get("record_type") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"chain": self.chain}
        if self.resnum is not None:
            out["resnum"] = self.resnum
        if self.atom is not None:
            out["atom"] = self.atom
        if self.atom_names:
            out["atom_names"] = list(self.atom_names)
        if self.elements:
            out["elements"] = list(self.elements)
        if self.n_atoms is not None:
            out["n_atoms"] = self.n_atoms
        if self.record_type is not None:
            out["record_type"] = self.record_type
        return out


@dataclass(frozen=True)
class GeometryConfig:
    """Configurable catalytic geometry definition.

    Defines two or three atom selections:
    - ``donor`` (or ``attack_candidates``): one or more candidate atoms; the
      nearest atom to ``electrophile`` is selected (deterministic tie-break).
    - ``electrophile`` (or ``center``): the single reference/target atom.
    - ``leaving`` (or ``angle_reference``, optional): a single reference atom
      defining the 3-atom angle ``donor_best -- electrophile -- leaving``.
      Omit for distance-only catalytic geometry evaluation.
    """

    donor: AtomSelector
    electrophile: AtomSelector
    leaving: AtomSelector | None = None
    clash_cutoff_A: float = 1.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeometryConfig":
        donor_raw = data.get("donor", data.get("attack_candidates"))
        elec_raw = data.get("electrophile", data.get("center"))
        leaving_raw = data.get("leaving", data.get("angle_reference"))
        if donor_raw is None or elec_raw is None:
            raise ValueError(
                "GeometryConfig requires 'donor'/'attack_candidates' and "
                "'electrophile'/'center' (plus optional 'leaving'/'angle_reference')."
            )
        donor = AtomSelector.from_dict(donor_raw)
        if "expected_attack_candidates" in data and donor.n_atoms is None:
            donor = AtomSelector(
                chain=donor.chain,
                resnum=donor.resnum,
                atom=donor.atom,
                atom_names=donor.atom_names,
                elements=donor.elements,
                n_atoms=int(data["expected_attack_candidates"]),
                record_type=donor.record_type,
            )
        return cls(
            donor=donor,
            electrophile=AtomSelector.from_dict(elec_raw),
            leaving=AtomSelector.from_dict(leaving_raw) if leaving_raw is not None else None,
            clash_cutoff_A=float(data.get("clash_cutoff_A", 1.5)),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "donor": self.donor.to_dict(),
            "electrophile": self.electrophile.to_dict(),
            "clash_cutoff_A": self.clash_cutoff_A,
        }
        if self.leaving is not None:
            out["leaving"] = self.leaving.to_dict()
        return out


@dataclass(frozen=True)
class ThresholdConfig:
    """Configurable pass/fail thresholds for catalytic geometry and confidence."""

    max_distance_A: float | None = 4.0
    min_angle_deg: float | None = 120.0
    max_angle_deg: float | None = 180.0
    min_confidence: float | None = None
    min_iptm: float | None = None
    min_ligand_iptm: float | None = None
    max_backbone_rmsd_A: float | None = 2.0
    max_motif_rmsd_A: float | None = 1.5
    max_clash_fraction: float | None = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ThresholdConfig":
        if not data:
            return cls()
        def _opt_float(key: str, default: float | None) -> float | None:
            if key not in data:
                return default
            val = data[key]
            return None if val is None else float(val)

        return cls(
            max_distance_A=_opt_float("max_distance_A", 4.0),
            min_angle_deg=_opt_float("min_angle_deg", 120.0),
            max_angle_deg=_opt_float("max_angle_deg", 180.0),
            min_confidence=_opt_float("min_confidence", None),
            min_iptm=_opt_float("min_iptm", None),
            min_ligand_iptm=_opt_float("min_ligand_iptm", None),
            max_backbone_rmsd_A=_opt_float("max_backbone_rmsd_A", 2.0),
            max_motif_rmsd_A=_opt_float("max_motif_rmsd_A", 1.5),
            max_clash_fraction=_opt_float("max_clash_fraction", 0.0),
        )


@dataclass(frozen=True)
class LigandSpec:
    """Non-polymer ligand/cofactor/ion entry for Boltz-2 co-folding inputs."""

    id: str
    smiles: str | None = None
    ccd: str | None = None

    def __post_init__(self) -> None:
        if bool(self.smiles) == bool(self.ccd):
            raise ValueError(
                f"LigandSpec({self.id!r}) must specify exactly one of 'smiles' or 'ccd'."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LigandSpec":
        return cls(
            id=str(data["id"]),
            smiles=data.get("smiles"),
            ccd=data.get("ccd", data.get("ccd_code")),
        )

    def to_boltz_entry(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id}
        if self.smiles:
            payload["smiles"] = self.smiles
        else:
            payload["ccd"] = self.ccd
        return {"ligand": payload}
