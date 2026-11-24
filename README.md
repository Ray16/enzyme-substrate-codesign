# Enzyme–Substrate Co-Design Workflows

Standalone, decoupled Python modules extracted from `Enzyme_agent` (`enzyme_design_agent` and `structural_modeling_agent`) for **saturation mutagenesis**, **partial diffusion**, **Boltz-2 co-folding**, **catalytic attack-geometry evaluation**, and **active-site contact analysis & visualization**.

## Architecture

```text
.
├── attack_geometry/          # Shared PDB/mmCIF parser, nucleophile-electrophile attack geometry,
│                             # proximity-weighted ensembles, Kabsch + ASP OD1/OD2 RMSD, Boltz-2 I/O
├── boltz_cofold/             # Standalone Boltz-2 co-folding: resumable MSA retrieval, standard &
│                             # FP32 serial-capped prediction, and post-fold QC verification
├── saturation_mutagenesis/   # Workflow 1: Whole-complex pocket scan -> single/multi-site variants
│                             # -> Boltz-2 co-folding -> attack-geometry & WT-delta ranking
├── partial_diffusion/        # Workflow 2: Partial diffusion (RFD4-Proteina / AlphaProtein Novo)
│                             # -> Boltz-2 refolding -> geometry + motif/backbone RMSD ranking
├── contact_analysis/         # Active-site substrate contact detection, ensemble frequency tables,
│                             # heavy_atom_donor_acceptor_v2 audit, and PyMOL + Matplotlib figures
├── examples/                 # Configurable YAML/JSON templates for every workflow
└── tests/                    # Offline, GPU-free unit & integration test suite
```

### Catalytic Geometry & Structural Scoring (`attack_geometry/`)

Both design workflows share a general, reaction-agnostic geometry and confidence evaluation engine:

- **Configurable catalytic geometry:** Supports user-defined atom selectors (`donor`, `electrophile`, and optional `leaving` / angle reference) to measure nucleophile–electrophile distances and optional 3-atom attack angles for any enzyme–substrate mechanism.
- **Ensemble statistics:** Computes standard ensemble summaries (`mean`, `median`, `harmonic_distance_A`) plus optional proximity-weighted angle statistics and wild-type bootstrap comparisons.
- **General symmetry-aware alignment & clash checks:** Proper-rotation Kabsch superposition for backbone and catalytic-motif RMSD (automatically resolving standard symmetric sidechain atom pairs across ASP, GLU, ARG, PHE, TYR, VAL, LEU, or custom pairs) and configurable protein–ligand clash screening.
- **Boltz-2 confidence metrics:** Extracts `iptm`, `ligand_iptm`, `complex_plddt`, `complex_iplddt`, `confidence_score`, and directional minimum `pair_chains_iptm` from `confidence_<id>_model_<k>.json`.

---

## Installation & Environment Notes

### Core package (CPU-only analysis, library generation, and dry runs)

```bash
pip install -e .
# With plotting & contact visualization support:
pip install -e ".[viz]"
```

Core dependencies are minimal (`numpy` and `PyYAML`, Python ≥ 3.9). No Biopython or MDAnalysis is required for PDB/mmCIF cofold analysis.

### Boltz-2 co-folding & PyMOL environment

To run live Boltz-2 co-folding predictions, install this package inside your GPU environment that provides the `boltz` CLI (e.g., an `enzymecage` or dedicated `boltz` conda environment with PyTorch + CUDA):

```bash
conda activate enzymecage   # or your Boltz-2 GPU environment
pip install -e ".[viz]"
boltz predict --help
```

- For 3D active-site contact snapshots, ensure `pymol` is on your `PATH` (or pass `--pymol-bin /path/to/pymol` / set `PYMOL_BIN`). If PyMOL is unavailable, `contact-analysis` automatically renders the contact-frequency bar chart and Markdown/CSV reports.
- For partial diffusion generation, install **RFD4-Proteina** (`rfd4-generate`) or **AlphaProtein Novo** (`apnovo-run`) in its dedicated environment and set `diffusion.executable` in your YAML config.

---

## Usage Examples

### 1. Standalone Boltz-2 Co-Folding (`boltz-cofold` / `python -m boltz_cofold`)

```bash
# Fetch & cache sequence-specific MSAs (batches <= 101, SHA-256 verified)
boltz-cofold prepare-msa \
  --fasta candidates.fasta \
  --out-dir msa_out \
  --cache-dir msa_cache

# Build Boltz-2 YAML inputs (enzyme + substrate/cofactor ligands)
boltz-cofold prepare-inputs \
  --fasta candidates.fasta \
  --config examples/cofold_config.yaml \
  --msa-index msa_out/index.tsv \
  --output-dir boltz_cofold_out

# Run Boltz-2 predictions (add --fp32 for V100/FP32 serial sample-cap mode, or --dry-run)
boltz-cofold predict \
  --inputs-dir boltz_cofold_out/boltz_inputs \
  --results-dir boltz_cofold_out/boltz_results \
  --config examples/cofold_config.yaml \
  --fp32

# Verify predicted models, sequences, and confidence JSONs
boltz-cofold verify \
  --results-dir boltz_cofold_out/boltz_results \
  --fasta candidates.fasta \
  --expected-samples 5 \
  --output-dir boltz_cofold_out/qc
```

---

### 2. Saturation Mutagenesis (`saturation-mutagenesis` / `python -m saturation_mutagenesis`)

```bash
# Step 1: Identify catalytic pocket residues around the connected holo complex
saturation-mutagenesis scan-pocket \
  --structure holo_scaffold.pdb \
  --cutoff 5.0 \
  --output pocket_positions.tsv

# Step 2: Generate single-site (or multi-site via --max-mutations 2) variants
saturation-mutagenesis generate-variants \
  --structure holo_scaffold.pdb \
  --positions pocket_positions.tsv \
  --label my_enzyme \
  --output-dir saturation_out

# Step 3–5: Prepare Boltz-2 inputs, co-fold, and score/rank variants (or run end-to-end)
saturation-mutagenesis run --config examples/saturation_config.yaml --dry-run
```

**Python API:**

```python
from saturation_mutagenesis import SaturationConfig, run_saturation_pipeline

cfg = SaturationConfig.from_file("examples/saturation_config.yaml")
summary = run_saturation_pipeline(cfg, run_boltz=True)
print(summary["evaluation"]["top_hits"][:5])
```

---

### 3. Partial Diffusion (`partial-diffusion` / `python -m partial_diffusion`)

```bash
# Run partial diffusion (RFD4-Proteina or AlphaProtein Novo) + Boltz-2 refold + ranking
partial-diffusion run --config examples/partial_diffusion_config.yaml --dry-run

# Or score pre-generated designs & Boltz-2 refolds directly:
partial-diffusion evaluate \
  --designs-dir partial_diffusion_out/diffusion_outputs \
  --results-dir partial_diffusion_out/boltz_results \
  --config examples/partial_diffusion_config.yaml \
  --output-dir partial_diffusion_out
```

**Python API:**

```python
from partial_diffusion import PartialDiffusionConfig, run_partial_diffusion_pipeline

cfg = PartialDiffusionConfig.from_file("examples/partial_diffusion_config.yaml")
summary = run_partial_diffusion_pipeline(cfg, run_diffusion=True, run_boltz=True)
print(summary["evaluation"]["top_hits"][:5])
```

---

### 4. Active-Site Contact Analysis & Visualization (`contact-analysis` / `python -m contact_analysis`)

```bash
# End-to-end ensemble contact frequency analysis + CSV/JSON/Markdown report + active-site figure
contact-analysis report \
  --name my_enzyme \
  --pose-dir boltz_results/my_enzyme \
  --output-dir contact_reports

# Run the strict heavy_atom_donor_acceptor_v2 audit (2.4 A quarantine, His/Cys separation)
contact-analysis audit \
  --pose model_0.pdb \
  --target-groups examples/contact_target_groups.json \
  --output contact_audit.json

# Plot ranked near-attack fraction & distance-vs-angle scatter from pose_geometry.csv
contact-analysis plot-geometry \
  --pose-geometry-csv saturation_out/pose_geometry.csv \
  --out saturation_out/attack_geometry_summary.png
```

---

### 5. Standalone Attack-Geometry Evaluation (`enzyme-geometry`)

```bash
enzyme-geometry model_0.pdb model_1.pdb \
  --config examples/saturation_config.yaml \
  --output geometry_report.json
```

---

## Standardized Outputs

- **Saturation mutagenesis:** `variant_summary.csv`, `pose_geometry.csv`, `top_hits.json`
- **Partial diffusion:** `design_summary.csv`, `pose_geometry.csv`, `top_hits.json`
- **Boltz-2 co-folding QC:** `cofold_verification.csv`, `cofold_verification.json`
- **Contact analysis & visualization:** `<name>_contact_frequency.json`, `<name>_contact_summary.csv`, `<name>_contact_report.md`, `<name>_contact_frequency.png`, and `attack_geometry_summary.png`

---

## Testing

```bash
pytest -q
```

All tests run offline and GPU-free using synthetic holo PDB/mmCIF structures and mock Boltz-2 outputs.

## License

MIT License — see [LICENSE](LICENSE).
