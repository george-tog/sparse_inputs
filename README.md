# odor_mix_code

Analysis code and bundled source data for
[Sparse input representations explain odor discrimination in complex, concentration-varying mixtures](https://www.biorxiv.org/content/10.64898/2026.01.27.702074v1).

Generated figures and newly generated result tables are ignored by default.

## Repository Layout

- `odor_mix_code/`: installable Python package with reusable model code and project path helpers.
- `scripts/`: notebooks used to generate manuscript figures, plus a compatibility import.
- `data/`: bundled source data from Zak 2020, Burton 2022, and Zak 2024.
- `results/GLUE/`: precomputed GLUE result table used when the optional GLUE package is unavailable.
- `figures/`: generated locally by notebooks; not tracked.
- `results/`: generated result tables; only the bundled GLUE fallback CSV is tracked.

## Setup

Create the mamba environment, install the local package, and register a notebook kernel:

```bash
mamba env create -f environment.yml
mamba activate sparse-inputs
pip install -e .
python -m ipykernel install --user --name sparse-inputs --display-name "Python (sparse-inputs)"
```

The notebooks write figures under `figures/` and generated result tables under `results/`. These directories are created as needed and are ignored by Git except for the tracked GLUE fallback result.

## Reproducing Figures

Run notebooks from the repository root or from the `scripts/` directory. The notebooks use package path helpers, so they should not depend on a machine-specific absolute path.

| Notebook | Manuscript output | Notes |
| --- | --- | --- |
| `scripts/1-all_decoding.ipynb` | Fig. 3B-F, Fig. S5 | Main decoding simulations. Some cells are longer-running because they sweep concentrations, backgrounds, and replicates. |
| `scripts/2-glue_analysis.ipynb` | Fig. 4 | Can plot from `results/GLUE/glomerular_poisson_gcmc_withshuffle_24bg_high_low_scale_0.0_10000points.csv` if the optional `gcmc` package is unavailable. |
| `scripts/3-OSN_PiC_data.ipynb` | Fig. S7B | One-vs-rest logistic regression on Zak 2024 OSN/PiC data and response sparsity analysis. |
| `scripts/4-sparse_expansive_code.ipynb` | Fig. S7A | Sparse expansive code comparison. Can load a prior generated result table if available. |
| `scripts/5-glom_data_sparsity.ipynb` | Fig. 3A, Fig. S6 | Glomerular model and data sparsity analysis using bundled Zak 2020 and Burton 2022 data. |

See `scripts/README.md` for the recommended run order, inputs, and outputs.

## Data

- `data/Zak_2020/Glomerular_Matrix.mat`: glomerular response data used in sparsity analyses.
- `data/Burton_2022/Fig1figsupp3source data 1_figS3_data.mat`: source data used in sparsity analyses.
- `data/Zak_2024/`: OSN and bouton response tables plus odor index used for OSN/PiC comparisons.
- `results/GLUE/glomerular_poisson_gcmc_withshuffle_24bg_high_low_scale_0.0_10000points.csv`: precomputed GLUE result table for reproducing Fig. 4 without running the optional GLUE pipeline.

## Reproducibility Notes

- Model initialization uses explicit seeds in the notebooks and helper functions.
- The installable module is `odor_mix_code.coding_model_fanofactor`.
- Project-relative paths are exposed from `odor_mix_code.paths`.
- Generated figures and non-bundled result tables are intentionally ignored to keep version control focused on source inputs and analysis code.
