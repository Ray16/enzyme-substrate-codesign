"""Command-line interface for standalone active-site contact analysis and visualization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attack_geometry.structure import load_structure

from .heavy_atom_audit import (
    Atom,
    ContactPolicy,
    TargetGroup,
    collapse_for_display,
    distinct_residue_count,
    evaluate_contacts,
)
from .substrate_contacts import analyze_ensemble, analyze_pose
from .visualize import plot_attack_geometry_summary, render_contact_figure, write_contact_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contact-analysis",
        description=(
            "Active-site substrate contact analysis, heavy-atom donor-acceptor audit, "
            "and publication-ready visualization for Boltz-2 cofolded structures."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. pose
    p_pose = sub.add_parser("pose", help="Detect active-site contacts in a single PDB/mmCIF structure.")
    p_pose.add_argument("--pose", type=Path, required=True, help="Input PDB or mmCIF file.")
    p_pose.add_argument("--protein-chain", default=None, help="Optional protein chain ID.")
    p_pose.add_argument("--substrate-chains", default="", help="Comma-separated substrate chain IDs.")
    p_pose.add_argument("--cofactor-chains", default="", help="Comma-separated cofactor chain IDs.")
    p_pose.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")

    # 2. ensemble
    p_ens = sub.add_parser("ensemble", help="Aggregate contact frequencies across a Boltz-2 pose ensemble.")
    p_ens.add_argument("--pose-dir", type=Path, default=None, help="Directory containing *.pdb / *.cif poses.")
    p_ens.add_argument("--poses-json", type=Path, default=None, help="JSON list of pose file paths.")
    p_ens.add_argument("--protein-chain", default=None, help="Optional protein chain ID.")
    p_ens.add_argument("--substrate-chains", default="", help="Comma-separated substrate chain IDs.")
    p_ens.add_argument("--cofactor-chains", default="", help="Comma-separated cofactor chain IDs.")
    p_ens.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")

    # 3. audit (heavy_atom_donor_acceptor_v2)
    p_aud = sub.add_parser("audit", help="Run strict heavy_atom_donor_acceptor_v2 contact audit.")
    p_aud.add_argument("--pose", type=Path, required=True, help="Input PDB or mmCIF structure.")
    p_aud.add_argument("--target-groups", type=Path, required=True, help="JSON list of TargetGroup specs.")
    p_aud.add_argument("--protein-chain", default="A", help="Protein chain ID (default: A).")
    p_aud.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")

    # 4. visualize
    p_viz = sub.add_parser("visualize", help="Render active-site 3D + contact-frequency bar chart.")
    p_viz.add_argument("--freq-json", type=Path, required=True, help="Output JSON from `ensemble`.")
    p_viz.add_argument("--pose", type=Path, default=None, help="Optional representative PDB/mmCIF for 3D PyMOL panel.")
    p_viz.add_argument("--title", default="Active-Site Substrate Contacts", help="Figure title.")
    p_viz.add_argument("--min-freq", type=float, default=0.5, help="Minimum contact frequency to highlight.")
    p_viz.add_argument("--pymol-bin", default=None, help="Optional path to PyMOL executable.")
    p_viz.add_argument("--out", type=Path, required=True, help="Output PNG path.")

    # 5. plot-geometry
    p_pg = sub.add_parser("plot-geometry", help="Plot attack-geometry & near-attack fraction summary from pose_geometry.csv.")
    p_pg.add_argument("--pose-geometry-csv", type=Path, required=True, help="pose_geometry.csv from saturation or partial diffusion.")
    p_pg.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    p_pg.add_argument("--near-distance", type=float, default=4.0, help="Near-attack distance cutoff in Å (default: 4.0).")
    p_pg.add_argument("--near-angle", type=float, default=150.0, help="Lower near-attack angle bound in deg (default: 150.0).")
    p_pg.add_argument("--max-angle", type=float, default=None, help="Optional upper near-attack angle bound in deg (e.g., 130 for Bürgi-Dunitz).")
    p_pg.add_argument("--target-angle", type=float, default=None, help="Optional target angle in deg (e.g., 107 for carbonyl addition, 180 for SN2).")
    p_pg.add_argument("--title", default="Catalytic Attack Geometry & Near-Attack Fraction", help="Figure title.")

    # 6. report (end-to-end)
    p_rep = sub.add_parser("report", help="Run ensemble contact analysis and write CSV/JSON/MD report + figure.")
    p_rep.add_argument("--name", required=True, help="Enzyme or design identifier.")
    p_rep.add_argument("--pose-dir", type=Path, default=None, help="Directory of Boltz-2 poses.")
    p_rep.add_argument("--poses-json", type=Path, default=None, help="JSON list of pose paths.")
    p_rep.add_argument("--representative-pose", type=Path, default=None, help="Pose for 3D panel (defaults to first pose).")
    p_rep.add_argument("--protein-chain", default=None, help="Optional protein chain ID.")
    p_rep.add_argument("--substrate-chains", default="", help="Comma-separated substrate chain IDs.")
    p_rep.add_argument("--cofactor-chains", default="", help="Comma-separated cofactor chain IDs.")
    p_rep.add_argument("--pymol-bin", default=None, help="Optional PyMOL executable path.")
    p_rep.add_argument("--no-figure", action="store_true", help="Skip figure rendering (tables & Markdown only).")
    p_rep.add_argument("--output-dir", type=Path, required=True, help="Output directory.")

    return parser


def _split_csv_arg(val: str) -> list[str] | None:
    items = [x.strip() for x in val.split(",") if x.strip()]
    return items or None


def _collect_poses(pose_dir: Path | None, poses_json: Path | None) -> list[Path]:
    if poses_json is not None:
        return [Path(p) for p in json.loads(poses_json.read_text())]
    if pose_dir is not None:
        paths = sorted(pose_dir.rglob("*.pdb")) + sorted(pose_dir.rglob("*.cif"))
        return paths
    raise ValueError("Provide either --pose-dir or --poses-json")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "pose":
        res = analyze_pose(
            args.pose,
            protein_chain=args.protein_chain,
            substrate_chains=_split_csv_arg(args.substrate_chains),
            cofactor_chains=_split_csv_arg(args.cofactor_chains),
        )
        text = json.dumps(res, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            sys.stdout.write(text)
        return 0

    if args.command == "ensemble":
        poses = _collect_poses(args.pose_dir, args.poses_json)
        res = analyze_ensemble(
            poses,
            protein_chain=args.protein_chain,
            substrate_chains=_split_csv_arg(args.substrate_chains),
            cofactor_chains=_split_csv_arg(args.cofactor_chains),
        )
        text = json.dumps(res, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            sys.stdout.write(text)
        return 0

    if args.command == "audit":
        st = load_structure(args.pose)
        atoms = [
            Atom(
                chain=a.chain,
                residue=a.resname.upper(),
                resi=str(a.resnum),
                name=a.name,
                element=a.element.upper(),
                xyz=(float(a.xyz[0]), float(a.xyz[1]), float(a.xyz[2])),
            )
            for a in st.atoms
        ]
        raw_groups = json.loads(args.target_groups.read_text())
        groups = [
            TargetGroup(
                group_id=str(g["group_id"]),
                chain=str(g["chain"]),
                resi=str(g["resi"]),
                acceptors=tuple(g["acceptors"]),
                formal_charge=int(g["formal_charge"]),
                provenance=str(g.get("provenance", "user_specified")),
            )
            for g in raw_groups
        ]
        policy = ContactPolicy()
        pairs = evaluate_contacts(atoms, groups, policy=policy, protein_chain=args.protein_chain)
        collapsed = collapse_for_display(pairs)
        payload = {
            "policy": policy.to_dict(),
            "distinct_contacting_residues": distinct_residue_count(pairs),
            "n_group_contacts": len(collapsed),
            "group_contacts": collapsed,
            "atom_pairs": pairs,
        }
        text = json.dumps(payload, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            sys.stdout.write(text)
        return 0

    if args.command == "visualize":
        freq = json.loads(args.freq_json.read_text())
        res = render_contact_figure(
            freq,
            args.out,
            pose_path=args.pose,
            title=args.title,
            min_freq=args.min_freq,
            pymol_bin=args.pymol_bin,
        )
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return 0

    if args.command == "plot-geometry":
        res = plot_attack_geometry_summary(
            args.pose_geometry_csv,
            args.out,
            near_distance_A=args.near_distance,
            near_angle_deg=args.near_angle,
            max_angle_deg=args.max_angle,
            target_angle_deg=args.target_angle,
            title=args.title,
        )
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return 0

    if args.command == "report":
        sub_chains = _split_csv_arg(args.substrate_chains)
        cof_chains = _split_csv_arg(args.cofactor_chains)
        poses = _collect_poses(args.pose_dir, args.poses_json)
        freq = analyze_ensemble(
            poses,
            protein_chain=args.protein_chain,
            substrate_chains=sub_chains,
            cofactor_chains=cof_chains,
        )
        rep_pose = args.representative_pose or (poses[0] if poses else None)
        res = write_contact_report(
            freq,
            args.output_dir,
            name=args.name,
            pose_path=rep_pose,
            pymol_bin=args.pymol_bin,
            render_figure=not args.no_figure,
            substrate_chains=sub_chains,
            cofactor_chains=cof_chains,
        )
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
