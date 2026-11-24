# Enzyme–Substrate Co-Design Workflows

Standalone, decoupled Python modules for **saturation mutagenesis** and **partial diffusion** enzyme design paired with **Boltz-2** co-folding validation and **catalytic attack-geometry** scoring.

## Architecture

```text
.
├── attack_geometry/          # Shared structure parser (PDB/mmCIF), attack-geometry metrics,
│                             # Kabsch + ASP OD1/OD2 symmetry RMSD, and Boltz-2 I/O
├── saturation_mutagenesis/   # Workflow 1: Whole-complex pocket scan -> single/multi-site
│                             # variants -> Boltz-2 co-folding -> geometry & WT-delta ranking
├── partial_diffusion/        # Workflow 2: Partial diffusion (RFD4-Proteina / AlphaProtein Novo)
│                             # -> Boltz-2 refolding -> attack geometry + motif/backbone RMSD ranking
├── examples/                 # Configurable YAML templates for both workflows
└── tests/                    # Offline, GPU-free unit & integration test suite
```

### Catalytic Attack-Geometry Metrics (`attack_geometry/`)

Both workflows share the same reaction-geometry evaluation engine:

- **Nucleophile–electrophile distance & attack angle:** Selects candidate nucleophilic atoms (`donor` / `attack_candidates`), picks the nearest heavy atom to `electrophile` (`center`), and computes the angle `donor_best – electrophile – leaving` in degrees.
- **Proximity-weighted ensemble aggregation:** Aggregates multi-pose Boltz-2 ensembles using inverse-distance weights $w_i = 1 / d_i$:
  - Weighted attack angle: $\bar{\theta}_w = \frac{\sum_i \theta_i / d_i}{\sum_i 1 / d_i}$
  - Harmonic distance: $\bar{d}_H = \frac{N}{\sum_i 1 / d_i}$
- **Bootstrap WT comparison (saturation mutagenesis):** Independent ensemble bootstrap resampling against matched wild-type poses (5th–95th percentile CIs and resampling improvement fractions).
- **Catalytic residue & backbone alignment (partial diffusion):** Proper-rotation Kabsch superposition for global `N/CA/C/O` backbone RMSD, symmetry-aware catalytic motif all-atom RMSD (resolving equivalent `ASP` `OD1`/`OD2` permutations), and protein–ligand steric clash fraction (`< 1.5 Å`).
- **Boltz-2 confidence metrics:** Extracts `iptm`, `ligand_iptm`, `complex_plddt`, `complex_iplddt`, `confidence_score`, and directional minimum `pair_chains_iptm` from `confidence_<id>_model_<k>.json`.

---

## Installation & Environment Notes

### Core package (CPU-only analysis, library generation, and dry runs)

```bash
pip install -e .
```

Core dependencies are minimal (`numpy` and `PyYAML`, Python ≥ 3.9). No Biopython is required.

### Boltz-2 co-folding environment (GPU)

To run live Boltz-2 co-folding predictions, install this package inside your GPU environment that provides the `boltz` CLI (e.g., an `enzymecage` or dedicated `boltz` conda environment with PyTorch + CUDA):

```bash
conda activate enzymecage   # or your Boltz-2 GPU environment
pip install -e .
# Verify boltz CLI is available:
boltz predict --help
```

For partial diffusion generation, install either **RFD4-Proteina** (`rfd4-generate`) or **AlphaProtein Novo** (`apnovo-run`) in its dedicated environment and point `diffusion.executable` in your YAML config to the corresponding executable.

---

## Usage Examples

### 1. Saturation Mutagenesis (`saturation-mutagenesis` / `python -m saturation_mutagenesis`)

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

# Step 3: Prepare Boltz-2 YAML inputs (enzyme + substrate/cofactor ligands)
saturation-mutagenesis prepare-boltz \
  --variants saturation_out/variants.tsv \
  --config examples/saturation_config.yaml \
  --output-dir saturation_out

# Step 4: Run Boltz-2 co-folding (add --dry-run to preview commands)
saturation-mutagenesis run-boltz \
  --inputs-dir saturation_out/boltz_inputs \
  --results-dir saturation_out/boltz_results \
  --config examples/saturation_config.yaml

# Step 5: Score & rank variants by attack geometry and Boltz-2 confidence
saturation-mutagenesis evaluate \
  --variants saturation_out/variants.tsv \
  --results-dir saturation_out/boltz_results \
  --config examples/saturation_config.yaml \
  --output-dir saturation_out

# Or run the full pipeline from a single config:
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

### 2. Partial Diffusion (`partial-diffusion` / `python -m partial_diffusion`)

```bash
# Step 1: Prepare / run partial diffusion generation (RFD4-Proteina or AlphaProtein Novo)
partial-diffusion run-diffusion \
  --config examples/partial_diffusion_config.yaml \
  --output-dir partial_diffusion_out/diffusion_outputs \
  --dry-run

# Step 2: Collect generated CIF/PDB designs and prepare Boltz-2 refolding inputs
partial-diffusion prepare-boltz \
  --designs-dir partial_diffusion_out/diffusion_outputs \
  --config examples/partial_diffusion_config.yaml \
  --output-dir partial_diffusion_out

# Step 3: Run unrestrained Boltz-2 co-folding validation
partial-diffusion run-boltz \
  --inputs-dir partial_diffusion_out/boltz_inputs \
  --results-dir partial_diffusion_out/boltz_results \
  --config examples/partial_diffusion_config.yaml

# Step 4: Score designs by attack geometry, motif/backbone RMSD, clashes, and confidence
partial-diffusion evaluate \
  --designs-dir partial_diffusion_out/diffusion_outputs \
  --results-dir partial_diffusion_out/boltz_results \
  --config examples/partial_diffusion_config.yaml \
  --output-dir partial_diffusion_out

# Or run end-to-end:
partial-diffusion run --config examples/partial_diffusion_config.yaml --dry-run
```

**Python API:**

```python
from partial_diffusion import PartialDiffusionConfig, run_partial_diffusion_pipeline

cfg = PartialDiffusionConfig.from_file("examples/partial_diffusion_config.yaml")
summary = run_partial_diffusion_pipeline(cfg, run_diffusion=True, run_boltz=True)
print(summary["evaluation"]["top_hits"][:5])
```

---

### 3. Standalone Attack-Geometry Evaluation (`enzyme-geometry`)

```bash
enzyme-geometry model_0.pdb model_1.pdb \
  --config examples/saturation_config.yaml \
  --output geometry_report.json
```

---

## Standardized Outputs

Both workflows emit standardized, machine-readable summaries:

- `variant_summary.csv` / `design_summary.csv`: Ranked table with sequences, ensemble attack geometry (`weighted_angle_deg`, `harmonic_distance_A`), WT deltas (saturation), backbone/motif RMSDs (partial diffusion), steric clash fraction, Boltz-2 confidence scores, and threshold pass/fail status.
- `pose_geometry.csv`: Per-pose nucleophile–electrophile distances, attack angles, clash fractions, and confidence metrics.
- `top_hits.json`: JSON summary of campaign counts and ranked top hits.

---

## Testing

```bash
pytest -q
```

All tests run offline and GPU-free using synthetic holo PDB/mmCIF structures and mock Boltz-2 outputs.

## License

MIT License — see [LICENSE](LICENSE).
