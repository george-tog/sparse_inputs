# Scripts

1-all_decoding.ipynb (Fig 3B-F, S5)

2-glue_analysis.ipynb (Fig4) - depends on ../results/GLUE/glomerular_gcmc_withshuffle_24bg_high_low_scale_0.0_10000points.csv if GLUE package is not public

3-OSN-PiC_data.ipynb (Fig S7B): one vs rest logistic regression and computes response sparsity 

4-sparse_expansive_code.ipynb (Fig S7A): Following Babadi and Sompolinsky, use a random expansive code from glomeruli to cortical neurons. We fit logistic regression instead of a Hebbian learning rule.

5-glom_data_sparsity.ipynb (3A, Fig S6): Computes Treves-Rolls sparsity from glomerular model at high and low concentrations, data collected in Zak et al. 2020 and Burton et al. 2022

coding_model_fanofactor.py: core glomerular response functions for Fano factor-scaled Gaussian noise. Adapted from Reddy et al. 2018 [https://github.com/greddy992/Odor-mixtures]