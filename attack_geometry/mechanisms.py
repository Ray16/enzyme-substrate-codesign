"""First-class enzyme reaction mechanism specifications and structural descriptor presets.

A reaction mechanism determines:
1. Which cofactors, metals, and protonation states must be modeled in Boltz-2.
2. Which primary and auxiliary structural descriptors are physically meaningful
   (e.g., inline ~180° attack angle for SN2/phosphoryl transfer vs. ~107°
   Bürgi-Dunitz trajectory + oxyanion-hole distances for carbonyl addition).
3. Sensible default thresholds and ranking criteria (all user-overridable).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import ThresholdConfig


@dataclass(frozen=True)
class MechanismSpec:
    """Declarative specification of an enzyme catalytic mechanism."""

    name: str
    description: str
    target_angle_deg: float | None
    default_rank_by: str
    default_thresholds: ThresholdConfig
    recommended_descriptors: tuple[str, ...]
    modeling_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "target_angle_deg": self.target_angle_deg,
            "default_rank_by": self.default_rank_by,
            "default_thresholds": asdict(self.default_thresholds),
            "recommended_descriptors": list(self.recommended_descriptors),
            "modeling_notes": self.modeling_notes,
        }


MECHANISMS: dict[str, MechanismSpec] = {
    "inline_sn2": MechanismSpec(
        name="inline_sn2",
        description=(
            "In-line nucleophilic displacement (e.g., phosphoryl transfer, "
            "adenylation, methyltransfer, haloalkane dehalogenation)."
        ),
        target_angle_deg=180.0,
        default_rank_by="angle_error",
        default_thresholds=ThresholdConfig(
            max_distance_A=4.0,
            min_angle_deg=120.0,
            max_angle_deg=180.0,
        ),
        recommended_descriptors=(
            "nucleophile_electrophile_distance_A",
            "inline_attack_angle_deg",
            "metal_coordination_distance_A",
            "clash_fraction",
        ),
        modeling_notes=(
            "Model the full reactive assembly including cofactor (e.g., ATP/SAM) "
            "and catalytic divalent metals (e.g., Mg2+/Mn2+) with appropriate formal charges."
        ),
    ),
    "carbonyl_addition": MechanismSpec(
        name="carbonyl_addition",
        description=(
            "Nucleophilic addition to a sp2 carbonyl/ester/amide carbon "
            "(e.g., serine/cysteine hydrolases, acyltransferases, proteases) "
            "following the Bürgi-Dunitz trajectory (~107 deg)."
        ),
        target_angle_deg=107.0,
        default_rank_by="angle_error",
        default_thresholds=ThresholdConfig(
            max_distance_A=3.8,
            min_angle_deg=85.0,
            max_angle_deg=130.0,
        ),
        recommended_descriptors=(
            "nucleophile_carbonyl_C_distance_A",
            "burgi_dunitz_angle_deg",
            "oxyanion_hole_distance_A",
            "catalytic_base_distance_A",
            "clash_fraction",
        ),
        modeling_notes=(
            "Select the catalytic nucleophile (e.g., Ser OG / Cys SG) as donor, "
            "substrate carbonyl carbon as electrophile, and carbonyl oxygen as "
            "angle_reference. Add oxyanion-hole and catalytic-base distances via extra_metrics."
        ),
    ),
    "general_acid_base": MechanismSpec(
        name="general_acid_base",
        description=(
            "Proton-transfer or general acid/base catalysis where heavy-atom "
            "donor-acceptor proximity governs preorganization (distance-primary)."
        ),
        target_angle_deg=None,
        default_rank_by="distance",
        default_thresholds=ThresholdConfig(
            max_distance_A=3.5,
            min_angle_deg=None,
            max_angle_deg=None,
        ),
        recommended_descriptors=(
            "acid_base_substrate_distance_A",
            "partner_residue_distance_A",
            "clash_fraction",
        ),
        modeling_notes=(
            "Specify the catalytic acid/base heteroatom as donor and the target "
            "substrate heavy atom as electrophile; omit leaving for distance-only scoring."
        ),
    ),
    "metalloenzyme_coordination": MechanismSpec(
        name="metalloenzyme_coordination",
        description=(
            "Inner-sphere metal-assisted catalysis (e.g., Zn2+/Fe2+/Mn2+ hydrolases, "
            "oxidoreductases) requiring direct substrate-metal coordination."
        ),
        target_angle_deg=None,
        default_rank_by="distance",
        default_thresholds=ThresholdConfig(
            max_distance_A=2.8,
            min_angle_deg=None,
            max_angle_deg=None,
        ),
        recommended_descriptors=(
            "substrate_metal_distance_A",
            "metal_ligand_shell_distances_A",
            "clash_fraction",
        ),
        modeling_notes=(
            "Include all catalytic metal ions in Boltz-2 inputs; verify inner-sphere "
            "coordination distances (typically 2.0-2.6 A) alongside protein ligand geometry."
        ),
    ),
    "custom": MechanismSpec(
        name="custom",
        description="User-defined reaction mechanism with explicit geometry, thresholds, and ranking.",
        target_angle_deg=None,
        default_rank_by="distance",
        default_thresholds=ThresholdConfig(
            max_distance_A=None,
            min_angle_deg=None,
            max_angle_deg=None,
        ),
        recommended_descriptors=(
            "distance_A",
            "angle_deg",
            "confidence_score_mean",
        ),
        modeling_notes="Provide explicit geometry selectors, thresholds, and rank_by in your config.",
    ),
}


def get_mechanism(name: str | None) -> MechanismSpec:
    """Look up a built-in ``MechanismSpec`` by name (defaults to ``'custom'``)."""
    if not name:
        return MECHANISMS["custom"]
    key = name.strip().lower()
    if key not in MECHANISMS:
        valid = ", ".join(sorted(MECHANISMS))
        raise ValueError(f"Unknown mechanism {name!r}; choose from: {valid}")
    return MECHANISMS[key]


def list_mechanisms() -> list[dict[str, Any]]:
    """Return serializable summaries of all registered mechanisms."""
    return [spec.to_dict() for spec in MECHANISMS.values()]


def merge_mechanism_thresholds(
    mechanism: MechanismSpec,
    user_thresholds: dict[str, Any] | None,
) -> ThresholdConfig:
    """Start from ``mechanism.default_thresholds`` and apply any explicit user overrides."""
    base = asdict(mechanism.default_thresholds)
    if user_thresholds:
        base.update(user_thresholds)
    return ThresholdConfig.from_dict(base)
