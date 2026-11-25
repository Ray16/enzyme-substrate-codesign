"""Active-site contact visualization (PyMOL + Matplotlib) and attack-geometry summary plots.

Decoupled from cluster-specific paths: detects ``pymol`` via ``PYMOL_BIN`` or
``shutil.which("pymol")``, and gracefully renders the contact-frequency bar
chart even when PyMOL is unavailable in a CPU-only environment.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CAT_COLOR: dict[str, str] = {
    "salt_bridge": "#1b7837",
    "h_bond": "#2166ac",
    "hydrophobic": "#999999",
}

DOCK_CUT_A = 5.0


def _top_residues(freq: dict[str, Any], min_freq: float = 0.5, max_res: int = 8) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for gk, g in freq.get("groups", {}).items():
        role = gk.split(":")[0]
        for c in g.get("contacts", ()):
            if c["category"] in ("hydrophobic", "backbone_hbond"):
                continue
            key = (c.get("chain", ""), c["res"])
            if key not in best or c["frequency"] > best[key]["frequency"]:
                best[key] = {
                    "chain": c.get("chain", ""),
                    "res": c["res"],
                    "category": c["category"],
                    "frequency": c["frequency"],
                    "role": role,
                }
    rows = [r for r in best.values() if r["frequency"] >= min_freq]
    rows.sort(key=lambda r: (-r["frequency"], r["res"]))
    return rows[:max_res]


def _resid(res: str) -> str:
    return "".join(ch for ch in res if ch.isdigit())


def _dockedness(
    freq: dict[str, Any],
    dock_cutoff_A: float = DOCK_CUT_A,
) -> tuple[float | None, bool, float | None]:
    react = next((g for k, g in freq.get("groups", {}).items() if k.startswith("reactive")), None)
    if react is None:
        return None, False, None
    d = react.get("d_to_center_A_mean")
    mg = react.get("metal_coordination_freq")
    return d, (d is not None and d <= dock_cutoff_A), mg


def _ligand_pymol_selections(
    pose_path: str | Path,
    *,
    substrate_chains: Sequence[str] | None = None,
    cofactor_chains: Sequence[str] | None = None,
) -> tuple[str, str, str]:
    """Build general PyMOL selection expressions for (cofactor, metals, substrate).

    Uses ``split_ligands`` on the representative pose so non-phosphorus cofactors
    (heme, flavin, NAD, PLP) and cofactor-free enzymes are handled accurately.
    """
    from attack_geometry.structure import load_structure
    from .substrate_contacts import split_ligands

    try:
        st = load_structure(pose_path)
        cof, metals, sub = split_ligands(
            st, substrate_chains=substrate_chains, cofactor_chains=cofactor_chains
        )
    except (ValueError, OSError):
        return (
            "byres (not polymer and elem P)",
            "not polymer and elem Mg+Mn+Zn+Fe+Ca+Ni+Co+Cu",
            "not polymer and not cofactor_sel and not metals_sel",
        )

    def _sel_expr(atoms) -> str:
        if not atoms:
            return "none"
        terms = sorted({f"(chain {a.chain} and resn {a.resname} and resi {a.resnum})" for a in atoms})
        return " or ".join(terms)

    return _sel_expr(cof), _sel_expr(metals), _sel_expr(sub)


def _find_pymol(pymol_bin: str | None = None) -> str | None:
    """Locate a PyMOL executable via explicit arg, PYMOL_BIN, PATH, or macOS app bundle."""
    if pymol_bin:
        return pymol_bin
    env = os.environ.get("PYMOL_BIN")
    if env:
        return env
    found = shutil.which("pymol")
    if found:
        return found
    for cand in (
        "/Applications/PyMOL.app/Contents/MacOS/PyMOL",
        "/opt/anaconda3/envs/pymol/bin/pymol",
        "/opt/homebrew/bin/pymol",
    ):
        if Path(cand).is_file():
            return cand
    return None


def _find_chimerax(chimerax_bin: str | None = None) -> str | None:
    """Locate a UCSF ChimeraX executable via explicit arg, CHIMERAX_BIN, PATH, or macOS app bundle."""
    if chimerax_bin:
        return chimerax_bin
    env = os.environ.get("CHIMERAX_BIN")
    if env:
        return env
    for name in ("chimerax", "ChimeraX"):
        found = shutil.which(name)
        if found:
            return found
    apps = sorted(Path("/Applications").glob("ChimeraX*.app/Contents/MacOS/ChimeraX"), reverse=True)
    if apps:
        return str(apps[0])
    return None


def _render_chimerax_active_site(
    pose_path: str | Path,
    residues: Sequence[dict[str, Any]],
    out_png: Path,
    chimerax_bin: str | None = None,
    substrate_chains: Sequence[str] | None = None,
    cofactor_chains: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """Render a headless UCSF ChimeraX active-site snapshot highlighting anchoring residues."""
    exe = _find_chimerax(chimerax_bin)
    if not exe:
        return False, "chimerax binary not found (set --chimerax-bin or CHIMERAX_BIN)"

    from attack_geometry.structure import load_structure
    from .substrate_contacts import split_ligands

    pose = str(Path(pose_path).expanduser().resolve())
    try:
        st = load_structure(pose)
        cof, metals, sub = split_ligands(
            st, substrate_chains=substrate_chains, cofactor_chains=cofactor_chains
        )
    except (ValueError, OSError) as exc:
        return False, str(exc)

    def _cx_sel(atoms) -> str:
        if not atoms:
            return ""
        terms = sorted({f"/{a.chain}:{a.resnum}" for a in atoms})
        return " | ".join(terms)

    sub_sel = _cx_sel(sub)
    cof_sel = _cx_sel(cof)
    met_sel = _cx_sel(metals)

    cxc = [
        f"open {pose}",
        "set bgColor white",
        "lighting soft",
        "graphics silhouettes true",
        "delete solvent",
        "hide atoms",
        "cartoon",
        "color polymer #d8d8d8",
        "transparency polymer 82 cartoons",
    ]
    focus_parts: list[str] = []
    if sub_sel:
        cxc.extend([f"show ({sub_sel})", f"style ({sub_sel}) stick", f"color ({sub_sel}) & C #e66101", f"color ({sub_sel}) byhetero"])
        focus_parts.append(f"({sub_sel})")
    if cof_sel:
        cxc.extend([f"show ({cof_sel})", f"style ({cof_sel}) stick", f"color ({cof_sel}) & C #555555", f"color ({cof_sel}) byhetero"])
        focus_parts.append(f"({cof_sel})")
    if met_sel:
        cxc.extend([f"show ({met_sel})", f"style ({met_sel}) sphere", f"color ({met_sel}) #1b7837"])
        focus_parts.append(f"({met_sel})")

    for r in residues:
        rid = _resid(r["res"])
        if not rid:
            continue
        ch = r.get("chain") or "A"
        sel = f"/{ch}:{rid}"
        col = CAT_COLOR.get(r["category"], "#666666")
        cxc.extend(
            [
                f"show {sel} & sidechain",
                f"style {sel} & sidechain stick",
                f"color {sel} & C {col}",
                f"color {sel} byhetero",
                f"label {sel}@CB text \"{r['res']}\" color black bgColor white height 0.9",
            ]
        )
        focus_parts.append(sel)

    if sub_sel and focus_parts:
        cxc.append(f"hbonds ({sub_sel}) restrict cross reveal true color #666666 radius 0.05")
    focus = " | ".join(focus_parts) if focus_parts else "ligand"
    cxc.extend(
        [
            f"view ({focus}) pad 0.25",
            f"save {out_png} width 1400 height 1100 supersample 3",
            "exit",
        ]
    )

    with tempfile.NamedTemporaryFile("w", suffix=".cxc", delete=False) as tmp:
        tmp.write("\n".join(cxc) + "\n")
        script_path = tmp.name
    try:
        proc = subprocess.run([exe, "--nogui", "--offscreen", script_path], capture_output=True, text=True, timeout=300)
        if not out_png.is_file():
            return False, proc.stderr[-500:] or f"chimerax exited with code {proc.returncode}"
        return True, ""
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass


def _render_pymol_active_site(
    pose_path: str | Path,
    residues: Sequence[dict[str, Any]],
    out_png: Path,
    pymol_bin: str | None = None,
    substrate_chains: Sequence[str] | None = None,
    cofactor_chains: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """Render a headless PyMOL active-site snapshot highlighting anchoring residues."""
    exe = _find_pymol(pymol_bin)
    if not exe:
        return False, "pymol binary not found (set --pymol-bin or PYMOL_BIN)"

    pose = str(Path(pose_path).expanduser().resolve())
    cof_sel, metal_sel, sub_sel = _ligand_pymol_selections(
        pose, substrate_chains=substrate_chains, cofactor_chains=cofactor_chains
    )
    pml = [
        f"load {pose}, m",
        "hide everything",
        "bg_color white",
        "remove solvent",
        "remove (resn WAT+HOH+SOL+TIP3+T3P+NA+CL+K)",
        "set orthoscopic, 1",
        "set ray_opaque_background, 1",
        "set ray_trace_mode, 1",
        "set ray_shadows, 0",
        "set antialias, 2",
        "set cartoon_transparency, 0.85",
        "set cartoon_side_chain_helper, 1",
        "set cartoon_fancy_helices, 1",
        "set stick_radius, 0.16",
        "set valence, 0",
        "set dash_color, gray40",
        "set dash_radius, 0.055",
        "set dash_gap, 0.35",
        "set dash_length, 0.25",
        f"select cofactor_sel, ({cof_sel})",
        f"select metals_sel, ({metal_sel})",
        f"select substrate_sel, ({sub_sel})",
        "show cartoon, polymer",
        "color gray85, polymer",
        "show sticks, substrate_sel",
        "color tv_orange, substrate_sel and elem C",
        "util.cnc substrate_sel",
        "set stick_radius, 0.26, substrate_sel",
        "show sticks, cofactor_sel",
        "color gray35, cofactor_sel and elem C",
        "util.cnc cofactor_sel",
        "show spheres, metals_sel",
        "color forest, metals_sel",
        "set sphere_scale, 0.45, metals_sel",
    ]
    rids: list[str] = []
    for idx_r, r in enumerate(residues):
        rid = _resid(r["res"])
        if not rid:
            continue
        chain_clause = f"chain {r['chain']} and " if r.get("chain") else ""
        sel_name = f"r_{idx_r}_{rid}"
        rids.append(sel_name)
        col = CAT_COLOR.get(r["category"], "#999999")
        rgb = [int(col[1:3], 16) / 255.0, int(col[3:5], 16) / 255.0, int(col[5:7], 16) / 255.0]
        pml.extend(
            [
                f"select {sel_name}, (polymer and {chain_clause}resi {rid} and not name N+C+O)",
                f"show sticks, {sel_name}",
                f"set_color c_{idx_r}_{rid}, [{rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f}]",
                f"color c_{idx_r}_{rid}, {sel_name} and elem C",
                f"util.cnc {sel_name}",
                f"distance dd_{idx_r}_{rid}, ({sel_name} and elem N+O+S), (substrate_sel and elem N+O+S), 4.0",
                f"hide labels, dd_{idx_r}_{rid}",
                f"label (polymer and {chain_clause}resi {rid} and name CB), '{r['res']}'",
            ]
        )
    site = "substrate_sel or metals_sel" + "".join(f" or {s}" for s in rids)
    pml.extend(
        [
            "set label_color, black",
            "set label_bg_color, white",
            "set label_bg_transparency, 0.2",
            "set label_font_id, 7",
            "set label_size, 18",
            "set label_position, [0, 0, 2.2]",
            f"select site, ({site})",
            "orient site",
            "zoom site, 2.2",
            f"png {out_png}, width=1400, height=1100, dpi=220, ray=1",
            "quit",
        ]
    )

    with tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False) as tmp:
        tmp.write("\n".join(pml) + "\n")
        script_path = tmp.name
    try:
        proc = subprocess.run([exe, "-cq", script_path], capture_output=True, text=True, timeout=300)
        if not out_png.is_file():
            return False, proc.stderr[-500:] or f"pymol exited with code {proc.returncode}"
        return True, ""
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass


def render_contact_figure(
    freq: dict[str, Any],
    out_png: str | Path,
    *,
    pose_path: str | Path | None = None,
    title: str = "Active-Site Substrate Contacts",
    min_freq: float = 0.5,
    dock_cutoff_A: float = DOCK_CUT_A,
    pymol_bin: str | None = None,
    chimerax_bin: str | None = None,
    renderer: str = "auto",
    substrate_chains: Sequence[str] | None = None,
    cofactor_chains: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Render paired 3D PyMOL/ChimeraX active-site view (when available) + contact-frequency bar chart."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for contact visualization (`pip install matplotlib`)."
        ) from exc

    out = Path(out_png).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    residues = _top_residues(freq, min_freq=min_freq)
    react_d, docked, react_mg = _dockedness(freq, dock_cutoff_A=dock_cutoff_A)
    if react_d is None:
        dock_txt, dock_col = "", "#666666"
    elif docked:
        dock_txt = f"reactive group -> reaction center: {react_d:.2f} A (within {dock_cutoff_A:.1f} A)"
        if react_mg is not None:
            dock_txt += f" | metal-coordinated in {react_mg * 100:.0f}% of poses"
        dock_col = "#1b7837"
    else:
        dock_txt = f"reactive group -> reaction center: {react_d:.2f} A (> {dock_cutoff_A:.1f} A)"
        dock_col = "#b23024"

    pymol_ok = False
    render_backend = "none"
    pymol_note = "no representative pose supplied"
    struct_png = out.with_name(out.stem + "_3d.png")
    if pose_path is not None:
        if renderer in ("auto", "pymol"):
            pymol_ok, pymol_note = _render_pymol_active_site(
                pose_path,
                residues,
                struct_png,
                pymol_bin=pymol_bin,
                substrate_chains=substrate_chains,
                cofactor_chains=cofactor_chains,
            )
            if pymol_ok:
                render_backend = "pymol"
        if not pymol_ok and renderer in ("auto", "chimerax"):
            cx_ok, cx_note = _render_chimerax_active_site(
                pose_path,
                residues,
                struct_png,
                chimerax_bin=chimerax_bin,
                substrate_chains=substrate_chains,
                cofactor_chains=cofactor_chains,
            )
            if cx_ok:
                pymol_ok = True
                render_backend = "chimerax"
                pymol_note = ""
            elif renderer == "chimerax":
                pymol_note = cx_note

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in residues:
        groups.setdefault(r["role"], []).append(r)
    roles = sorted(groups, key=lambda k: {"reactive": 0, "secondary": 1}.get(k, 9))

    if pymol_ok:
        fig = plt.figure(figsize=(14, 6.2))
        outer = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.05], wspace=0.18)
        ax_left = fig.add_subplot(outer[0])
        ax_left.imshow(plt.imread(struct_png))
        ax_left.axis("off")
        ax_left.set_title("Active-site structure & contacting residues", fontsize=11, fontweight="bold", pad=6)
        right_slot = outer[1]
    else:
        fig = plt.figure(figsize=(9.5, 5.2))
        outer = fig.add_gridspec(1, 1)
        right_slot = outer[0]

    if not roles:
        ax = fig.add_subplot(right_slot)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"No persistent side-chain anchor (frequency >= {min_freq:.2f})",
            ha="center",
            va="center",
            fontsize=12,
            color="#b23024",
            fontweight="bold",
        )
    else:
        rgs = right_slot.subgridspec(1, len(roles), width_ratios=[max(1, len(groups[r])) for r in roles], wspace=0.2)
        axes = []
        section = {"reactive": "Reactive end", "secondary": "Distal / secondary end"}
        for idx, role in enumerate(roles):
            ax = fig.add_subplot(rgs[idx], sharey=axes[0] if axes else None)
            axes.append(ax)
            rows = sorted(groups[role], key=lambda r: -r["frequency"])
            xp = np.arange(len(rows))
            cols = [CAT_COLOR.get(r["category"], "#999999") for r in rows]
            ax.bar(xp, [r["frequency"] for r in rows], color=cols, edgecolor="black", linewidth=0.6, width=0.68)
            for x, r in zip(xp, rows):
                ax.text(x, r["frequency"] + 0.02, f"{r['frequency']:.2f}", ha="center", fontsize=9, fontweight="bold")
            ax.set_xticks(xp)
            ax.set_xticklabels([r["res"] for r in rows], fontsize=10, rotation=20, ha="right")
            ax.set_ylim(0, 1.18)
            ax.set_title(section.get(role, role), fontsize=11, fontweight="bold")
            if idx == 0:
                ax.set_ylabel("Contact frequency", fontsize=11)
            else:
                ax.tick_params(labelleft=False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

        cats = sorted({r["category"] for r in residues})
        handles = [Patch(facecolor=CAT_COLOR[c], edgecolor="black", label=c.replace("_", " ")) for c in cats]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(handles),
            fontsize=9.5,
            frameon=False,
        )

    fig.subplots_adjust(top=0.80 if dock_txt else 0.87, bottom=0.18)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    if dock_txt:
        fig.text(0.5, 0.90, dock_txt, ha="center", va="top", fontsize=10.5, color=dock_col, fontweight="bold")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "figure_path": str(out),
        "pymol_rendered": pymol_ok,
        "pymol_note": pymol_note,
        "n_highlighted_residues": len(residues),
    }


def plot_attack_geometry_summary(
    pose_geometry_csv: str | Path,
    out_png: str | Path,
    *,
    id_column: str | None = None,
    near_distance_A: float = 4.0,
    near_angle_deg: float = 150.0,
    min_angle_deg: float | None = None,
    max_angle_deg: float | None = None,
    target_angle_deg: float | None = None,
    top_n: int = 25,
    title: str = "Catalytic Attack Geometry & Near-Attack Fraction",
) -> dict[str, Any]:
    """Plot per-variant/design near-attack fraction and distance(-vs-angle) distribution.

    Supports both angle-constrained mechanisms (e.g., ``inline_sn2`` with
    ``min_angle_deg=150``, or ``carbonyl_addition`` with ``min_angle_deg=85,
    max_angle_deg=130, target_angle_deg=107``) and distance-only mechanisms
    (e.g., ``general_acid_base``, ``metalloenzyme_coordination``) where
    ``angle_deg`` is absent or empty.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for geometry plotting (`pip install matplotlib`)."
        ) from exc

    rows: list[dict[str, str]] = []
    with Path(pose_geometry_csv).expanduser().open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {pose_geometry_csv}")

    id_col = id_column or ("variant_id" if "variant_id" in fields else "design_id")
    has_angles = "angle_deg" in fields and all(
        r.get("angle_deg") not in (None, "", "None") for r in rows
    )
    eff_min_angle = min_angle_deg if min_angle_deg is not None else near_angle_deg
    eff_max_angle = max_angle_deg if max_angle_deg is not None else 180.0

    by_id: dict[str, list[tuple[float, float | None]]] = {}
    for r in rows:
        d_val = float(r["distance_A"])
        a_val = float(r["angle_deg"]) if has_angles else None
        by_id.setdefault(r[id_col], []).append((d_val, a_val))

    stats = []
    for ident, pairs in by_id.items():
        d = np.asarray([p[0] for p in pairs], dtype=float)
        h_dist = float(len(d) / np.sum(1.0 / d))
        if has_angles:
            a = np.asarray([float(p[1]) for p in pairs], dtype=float)
            nac = float(np.mean((d <= near_distance_A) & (a >= eff_min_angle) & (a <= eff_max_angle)))
            w_ang = float(np.sum(a / d) / np.sum(1.0 / d))
            ang_score = -abs(w_ang - target_angle_deg) if target_angle_deg is not None else w_ang
            stats.append(
                {
                    "id": ident,
                    "nac": nac,
                    "weighted_angle_deg": w_ang,
                    "angle_score": ang_score,
                    "harmonic_distance_A": h_dist,
                    "n": len(d),
                }
            )
        else:
            nac = float(np.mean(d <= near_distance_A))
            stats.append(
                {
                    "id": ident,
                    "nac": nac,
                    "angle_score": 0.0,
                    "harmonic_distance_A": h_dist,
                    "n": len(d),
                }
            )

    stats.sort(key=lambda s: (s["nac"], s["angle_score"], -s["harmonic_distance_A"]))
    shown = stats[-top_n:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4.2, 0.34 * len(shown) + 1.8)))
    y = np.arange(len(shown))
    colors = ["#1b7837" if s["nac"] >= 0.5 else "#999999" for s in shown]
    ax1.barh(y, [s["nac"] for s in shown], color=colors, edgecolor="white")
    ax1.set_yticks(y)
    ax1.set_yticklabels([s["id"] for s in shown], fontsize=9)
    ax1.set_xlim(0, 1.05)
    if has_angles:
        ang_lbl = (
            f"{eff_min_angle:.0f}–{eff_max_angle:.0f} deg"
            if max_angle_deg is not None
            else f"> {eff_min_angle:.0f} deg"
        )
        ax1.set_xlabel(f"Near-attack fraction (d <= {near_distance_A:.1f} A & angle {ang_lbl})")
    else:
        ax1.set_xlabel(f"Near-attack fraction (d <= {near_distance_A:.1f} A)")
    ax1.set_title("Ranked near-attack fraction")
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    all_d = [float(r["distance_A"]) for r in rows]
    if has_angles:
        all_a = [float(r["angle_deg"]) for r in rows]
        ax2.scatter(all_d, all_a, alpha=0.65, s=26, color="#2166ac", edgecolors="white", linewidth=0.4)
        ax2.axvline(near_distance_A, ls="--", color="#b23024", lw=1.0)
        ax2.axhline(eff_min_angle, ls="--", color="#b23024", lw=1.0)
        if max_angle_deg is not None:
            ax2.axhline(eff_max_angle, ls="--", color="#b23024", lw=1.0)
        if target_angle_deg is not None:
            ax2.axhline(target_angle_deg, ls=":", color="#1b7837", lw=1.2)
        ax2.set_xlabel("Catalytic distance (A)")
        ax2.set_ylabel("Attack angle (deg)")
        ax2.set_title("Per-pose attack geometry")
    else:
        ax2.hist(all_d, bins=min(20, max(5, len(all_d) // 2)), color="#2166ac", edgecolor="white", alpha=0.8)
        ax2.axvline(near_distance_A, ls="--", color="#b23024", lw=1.2)
        ax2.set_xlabel("Catalytic distance (A)")
        ax2.set_ylabel("Pose count")
        ax2.set_title("Per-pose catalytic distance distribution")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = Path(out_png).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"figure_path": str(out), "n_candidates": len(stats), "top_candidate": stats[-1]}


def write_contact_report(
    freq: dict[str, Any],
    output_dir: str | Path,
    *,
    name: str = "enzyme",
    pose_path: str | Path | None = None,
    min_freq: float = 0.5,
    pymol_bin: str | None = None,
    chimerax_bin: str | None = None,
    renderer: str = "auto",
    render_figure: bool = True,
    substrate_chains: Sequence[str] | None = None,
    cofactor_chains: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export standardized JSON, CSV, Markdown, and PNG contact analysis artifacts."""
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{name}_contact_frequency.json"
    json_path.write_text(json.dumps(freq, indent=2) + "\n")

    csv_rows: list[dict[str, Any]] = []
    for gk, g in freq.get("groups", {}).items():
        role, kind = gk.split(":", 1) if ":" in gk else (gk, "")
        for c in g.get("contacts", ()):
            csv_rows.append(
                {
                    "name": name,
                    "role": role,
                    "group_kind": kind,
                    "residue": c["res"],
                    "category": c["category"],
                    "frequency": c["frequency"],
                    "count": c.get("count", ""),
                    "metal_coordination_freq": g.get("metal_coordination_freq", ""),
                    "d_to_center_A_mean": g.get("d_to_center_A_mean", ""),
                }
            )
    csv_path = out / f"{name}_contact_summary.csv"
    if csv_rows:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    else:
        csv_path.write_text("name,role,group_kind,residue,category,frequency,count,metal_coordination_freq,d_to_center_A_mean\n")

    md_lines = [
        f"# Active-Site Contact Analysis: {name}",
        "",
        f"- **Structures analyzed:** {freq.get('n_structures', 0)} (failed: {freq.get('n_failed', 0)})",
        "",
        "| Role | Group | Residue | Category | Frequency |",
        "|---|---|---|---|---:|",
    ]
    for r in csv_rows:
        md_lines.append(
            f"| {r['role']} | {r['group_kind']} | {r['residue']} | {r['category']} | {r['frequency']:.3f} |"
        )
    if not csv_rows:
        md_lines.append("| — | — | No contacts detected | — | — |")
    md_path = out / f"{name}_contact_report.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    fig_info = None
    if render_figure:
        try:
            fig_info = render_contact_figure(
                freq,
                out / f"{name}_contact_frequency.png",
                pose_path=pose_path,
                title=f"{name} — Active-Site Substrate Contacts",
                min_freq=min_freq,
                pymol_bin=pymol_bin,
                chimerax_bin=chimerax_bin,
                renderer=renderer,
                substrate_chains=substrate_chains,
                cofactor_chains=cofactor_chains,
            )
        except ImportError as exc:
            fig_info = {"skipped": str(exc)}

    return {
        "name": name,
        "contact_frequency_json": str(json_path),
        "contact_summary_csv": str(csv_path),
        "contact_report_md": str(md_path),
        "figure": fig_info,
    }
