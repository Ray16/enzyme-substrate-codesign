"""Generate colleague-ready example visualizations demonstrating core workflows.

Produces three figures in ``--out-dir`` (default: ``docs/images``):
1. ``sn2_saturation_geometry.png`` — In-line SN2 saturation mutagenesis near-attack
   fraction & distance-vs-angle scatter (target 180 deg).
2. ``hydrolase_burgi_dunitz_geometry.png`` — Serine hydrolase carbonyl-addition
   saturation scan with Bürgi-Dunitz angle window (85-130 deg, target 107 deg).
3. ``active_site_contacts_example.png`` — Ensemble active-site substrate contact
   frequencies across reactive and distal functional groups.

Run:
    python examples/generate_example_figures.py --out-dir docs/images
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contact_analysis.visualize import plot_attack_geometry_summary, render_contact_figure


def _generate_sn2_figure(out_png: Path, work_dir: Path) -> dict:
    """Simulate a 10-variant inline_sn2 saturation scan (5 Boltz-2 poses each)."""
    rng = np.random.default_rng(42)
    variants = [
        ("WT", 3.9, 0.35, 146.0, 9.0),
        ("D143E", 3.1, 0.18, 168.0, 5.0),
        ("S190T", 3.25, 0.20, 162.0, 6.0),
        ("D143E_S190T", 2.95, 0.15, 172.0, 4.0),
        ("L164I", 3.45, 0.25, 156.0, 7.0),
        ("K215R", 3.55, 0.28, 152.0, 8.0),
        ("V188A", 4.15, 0.35, 138.0, 10.0),
        ("F201W", 4.40, 0.40, 128.0, 12.0),
        ("N118Q", 3.65, 0.25, 151.0, 7.5),
        ("D143A_ctrl", 5.20, 0.45, 105.0, 14.0),
    ]
    csv_path = work_dir / "sn2_pose_geometry.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant_id", "sample", "distance_A", "angle_deg"])
        w.writeheader()
        for vid, d_mu, d_sd, a_mu, a_sd in variants:
            for sample in range(5):
                d = float(np.clip(rng.normal(d_mu, d_sd), 2.5, 6.5))
                a = float(np.clip(rng.normal(a_mu, a_sd), 60.0, 179.5))
                w.writerow({"variant_id": vid, "sample": sample, "distance_A": round(d, 3), "angle_deg": round(a, 2)})

    return plot_attack_geometry_summary(
        csv_path,
        out_png,
        near_distance_A=4.0,
        min_angle_deg=150.0,
        target_angle_deg=180.0,
        title="In-Line SN2 Saturation Scan — Attack Geometry & Near-Attack Fraction",
    )


def _generate_hydrolase_figure(out_png: Path, work_dir: Path) -> dict:
    """Simulate a serine hydrolase carbonyl_addition scan (Bürgi-Dunitz ~107 deg)."""
    rng = np.random.default_rng(7)
    variants = [
        ("WT", 3.7, 0.30, 126.0, 12.0),
        ("G52A", 3.15, 0.16, 107.5, 5.5),
        ("L155V", 3.25, 0.20, 109.0, 6.5),
        ("G52A_L155V", 2.98, 0.14, 106.5, 4.5),
        ("M198I", 3.40, 0.22, 113.0, 7.5),
        ("Y124F", 3.55, 0.28, 119.0, 9.0),
        ("W160F", 3.90, 0.32, 134.0, 11.0),
        ("H220N_ctrl", 4.60, 0.40, 152.0, 10.0),
        ("S120C", 3.30, 0.18, 105.0, 6.0),
    ]
    csv_path = work_dir / "hydrolase_pose_geometry.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant_id", "sample", "distance_A", "angle_deg"])
        w.writeheader()
        for vid, d_mu, d_sd, a_mu, a_sd in variants:
            for sample in range(5):
                d = float(np.clip(rng.normal(d_mu, d_sd), 2.5, 5.8))
                a = float(np.clip(rng.normal(a_mu, a_sd), 70.0, 175.0))
                w.writerow({"variant_id": vid, "sample": sample, "distance_A": round(d, 3), "angle_deg": round(a, 2)})

    return plot_attack_geometry_summary(
        csv_path,
        out_png,
        near_distance_A=3.8,
        min_angle_deg=85.0,
        max_angle_deg=130.0,
        target_angle_deg=107.0,
        title="Serine Hydrolase (Bürgi–Dunitz ~107°) — Saturation Geometry Summary",
    )


def _generate_contacts_figure(out_png: Path) -> dict:
    """Render an example ensemble active-site contact frequency figure."""
    freq = {
        "source": "boltz_pose_ensemble",
        "n_structures": 20,
        "n_failed": 0,
        "groups": {
            "reactive:carboxylate": {
                "contacts": [
                    {"chain": "A", "res": "LYS143", "category": "salt_bridge", "frequency": 0.95, "count": 19},
                    {"chain": "A", "res": "SER190", "category": "h_bond", "frequency": 0.90, "count": 18},
                    {"chain": "A", "res": "THR192", "category": "h_bond", "frequency": 0.75, "count": 15},
                    {"chain": "A", "res": "HIS211", "category": "h_bond", "frequency": 0.60, "count": 12},
                ],
                "metal_coordination_freq": 0.90,
                "d_to_center_A_mean": 3.12,
                "d_to_center_A_std": 0.19,
            },
            "secondary:carboxylate": {
                "contacts": [
                    {"chain": "A", "res": "ARG215", "category": "salt_bridge", "frequency": 1.00, "count": 20},
                    {"chain": "A", "res": "TYR244", "category": "h_bond", "frequency": 0.85, "count": 17},
                    {"chain": "A", "res": "ASN118", "category": "h_bond", "frequency": 0.65, "count": 13},
                ],
                "metal_coordination_freq": 0.0,
                "d_to_center_A_mean": 7.45,
                "d_to_center_A_std": 0.34,
            },
        },
    }
    return render_contact_figure(
        freq,
        out_png,
        title="D143E_S190T — Ensemble Active-Site Substrate Contacts (n = 20 poses)",
        min_freq=0.5,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        r1 = _generate_sn2_figure(args.out_dir / "sn2_saturation_geometry.png", tmp_dir)
        r2 = _generate_hydrolase_figure(args.out_dir / "hydrolase_burgi_dunitz_geometry.png", tmp_dir)
        r3 = _generate_contacts_figure(args.out_dir / "active_site_contacts_example.png")

    print("Generated example figures:")
    for info in (r1, r2, r3):
        print(" -", info["figure_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
