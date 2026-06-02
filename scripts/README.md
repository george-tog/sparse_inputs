# Notebook Execution Guide

Run these notebooks with the `odor-mix-code` environment described in the top-level `README.md`. They use `odor_mix_code.paths` for project-relative `data/`, `results/`, and `figures/` paths.

## Recommended Order

1. `5-glom_data_sparsity.ipynb`
   - Generates Fig. 3A and Fig. S6.
   - Inputs: `data/Zak_2020/Glomerular_Matrix.mat`, `data/Burton_2022/Fig1figsupp3source data 1_figS3_data.mat`.
   - Outputs: glomerular sparsity figures under `figures/manuscript/`.

2. `1-all_decoding.ipynb`
   - Generates Fig. 3B-F and Fig. S5.
   - Inputs: model simulations from `odor_mix_code.coding_model_fanofactor`.
   - Outputs: decoding result CSVs under `results/` and figures under `figures/`.
   - Runtime note: this is the main simulation notebook and includes longer concentration/background/replicate sweeps. Existing CSVs are loaded for plotting by default; set `FORCE_RECOMPUTE = True` in the notebook to regenerate all decoding results.

3. `2-glue_analysis.ipynb`
   - Generates Fig. 4.
   - Inputs: either the optional `gcmc` package for recomputation or the bundled fallback table:
     `results/GLUE/glomerular_poisson_gcmc_withshuffle_24bg_high_low_scale_0.0_10000points.csv`.
   - Outputs: GLUE capacity/dimension/radius figures under `figures/`.

4. `4-sparse_expansive_code.ipynb`
   - Generates Fig. S7A.
   - Inputs: model simulations and, when `load_prior_result = True`, a previously generated
     `results/OB_PiC_comparison/expansive_code_poisson.csv`.
   - Outputs: sparse expansive code result CSVs under `results/OB_PiC_comparison/` and figures under `figures/manuscript/`.

5. `3-OSN_PiC_data.ipynb`
   - Generates Fig. S7B and computes response sparsity for Zak 2024 data.
   - Inputs: `data/Zak_2024/`.
   - Outputs: OSN/PiC decodability figures under `figures/manuscript/`.

## GLUE Fallback

The GLUE recomputation path imports `gcmc`, which may not be publicly installable in every environment. If `gcmc` is unavailable, skip the recomputation cell and run the plotting cells from the bundled CSV in `results/GLUE/`.

## Generated Outputs

Generated figures and newly generated results are ignored by Git. The only tracked file under `results/` is the bundled GLUE fallback CSV used for reviewer reproduction.

## Compatibility Module

`scripts/coding_model_fanofactor.py` is a shim for notebooks launched from the `scripts/` directory. New code should import from:

```python
from odor_mix_code.coding_model_fanofactor import *
```
