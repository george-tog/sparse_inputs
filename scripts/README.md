# Scripts

1-all_decoding.ipynb (Fig 4C-E, S6, S7)

2-glue_analysis.ipynb (Fig3) - depends on ../results/GLUE/glomerular_gcmc_withshuffle_24bg_high_low_scale_0.0_10000points.csv if GLUE package is not public

3-glom_model_sparsity.ipynb (Fig 4A, S8B, S8C): Computes Treves-Rolls sparsity from glomerular model at high and low concentrations

4-OSN-PiC_data.ipynb (Fig S9B): one vs rest logistic regression and computes response sparsity 

5-sparse_expansive_code.ipynb (Fig S9A): Following Babadi and Sompolinsky, use a random expansive code from glomeruli to cortical neurons. We fit logistic regression instead of a Hebbian learning rule

6-glom_data_sparsity.ipynb (Fig S8A): Computes Treves-Rolls sparsity from data collected in Zak et al. 2020 

coding_model_fanofactor.py: core functionality for fano factor-scaled Gaussian noise