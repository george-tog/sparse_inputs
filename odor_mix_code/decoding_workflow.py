from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from odor_mix_code.coding_model_fanofactor import (
    init_model_parameters,
    plot_matched_conc_performance_curves,
    plot_performance_curves,
    plot_psychometric_curve,
    psychometric_summary,
    run_decoder_experiments_k_datasets,
)
from odor_mix_code.paths import FIGURES_DIR, RESULTS_DIR


def _default_cue_concs() -> np.ndarray:
    return np.logspace(-7, -3, 9)


@dataclass(frozen=True)
class DecoderCombo:
    rho: float
    bg_conc: float
    train_cue_conc: Any
    penalty: str
    F_max: float
    fano_factor: float


@dataclass
class DecoderSweepConfig:
    label: str
    cue_concs: Any = field(default_factory=_default_cue_concs)
    train_cue_concs: list[Any] = field(default_factory=lambda: [1e-2])
    bg_concs: list[float] = field(default_factory=lambda: [10 ** (-2.5)])
    rhos: list[float] = field(default_factory=lambda: [0.5])
    penalties: list[str] = field(default_factory=lambda: ["l1"])
    F_maxes: list[float] = field(default_factory=lambda: [10.0])
    fano_factors: list[float] = field(default_factory=lambda: [1.0])
    n_train: int = 1000
    n_test: int = 1000
    N: int = 20
    total_bg: int = 16
    n_replicates: int = 10
    k: int = 10
    seed: int = 42
    n_jobs: int = 12
    force_recompute: bool = False
    make_figures: bool = True
    show_plots: bool = True
    cue: int = -1
    n_bg: int = -1
    clf_params: dict[str, Any] = field(default_factory=lambda: {
        "penalty": "l1",
        "C": 1.0,
        "max_iter": 1000,
        "solver": "liblinear",
    })
    results_dir: Path = RESULTS_DIR
    figures_dir: Path = FIGURES_DIR


@dataclass
class DecoderSweepResult:
    config: DecoderSweepConfig
    summaries: list[dict[str, Any]]
    result_df: pd.DataFrame
    training_df: pd.DataFrame | None = None


def train_conc_tag(train_cue_conc: Any, csv: bool = False) -> str:
    if np.isscalar(train_cue_conc):
        return f"trainconc_{float(train_cue_conc):.2e}"
    if csv:
        return "trainconc_allconcs"
    arr = np.asarray(train_cue_conc).reshape(-1)
    return f"trainconc_all_{arr[0]:.2e}_to_{arr[-1]:.2e}_n{arr.size}"


def train_conc_metadata(train_cue_conc: Any) -> float | str:
    if np.isscalar(train_cue_conc):
        return float(train_cue_conc)
    return "allconcs"


def csv_noise_desc(combo: DecoderCombo) -> str:
    return f"noise_fano_{combo.fano_factor:.2e}"


def figure_noise_desc(combo: DecoderCombo) -> str:
    return f"noise_fano_{combo.fano_factor:.2e}"


def iter_combos(config: DecoderSweepConfig) -> list[DecoderCombo]:
    return [
        DecoderCombo(
            rho=float(rho),
            bg_conc=float(bg_conc),
            train_cue_conc=train_cue_conc,
            penalty=penalty,
            F_max=float(F_max),
            fano_factor=float(fano_factor),
        )
        for rho, bg_conc, train_cue_conc, penalty, F_max, fano_factor
        in product(
            config.rhos,
            config.bg_concs,
            config.train_cue_concs,
            config.penalties,
            config.F_maxes,
            config.fano_factors,
        )
    ]


def make_model_params(config: DecoderSweepConfig, combo: DecoderCombo) -> dict[str, Any]:
    return init_model_parameters(
        total_bg=config.total_bg,
        N=config.N,
        rho=combo.rho,
        train_cue_conc=combo.train_cue_conc,
        bg_conc=combo.bg_conc,
        F_max=combo.F_max,
        n=config.n_replicates,
        seed=config.seed,
        fano_factor=combo.fano_factor,
    )


def result_paths(config: DecoderSweepConfig, combo: DecoderCombo) -> dict[str, Path]:
    model_params = make_model_params(config, combo)
    base_dir = config.results_dir / "target_bg_conc_noise_grid" / combo.penalty
    core = (
        f"{train_conc_tag(combo.train_cue_conc, csv=True)}_"
        f"bgconc_{model_params['bg_conc']:.2e}_"
        f"rho_{model_params['rho']}_"
        f"receptors_{model_params['N'] ** 2}_"
        f"totalbg_{model_params['total_bg']}_"
        f"Fmax_{combo.F_max}_{csv_noise_desc(combo)}_"
        f"n_reps_{config.k}"
    )
    return {
        "performance": base_dir / f"performance_{combo.penalty}_{core}.csv",
        "training": base_dir / f"training_performance_{combo.penalty}_{core}.csv",
        "weights": base_dir / f"decoder_{combo.penalty}_weights_{core}.csv",
    }


def figure_paths(config: DecoderSweepConfig, combo: DecoderCombo) -> dict[str, Path]:
    model_params = make_model_params(config, combo)
    base_dir = config.figures_dir / combo.penalty
    core = (
        f"{train_conc_tag(combo.train_cue_conc)}_"
        f"bgconc_{model_params['bg_conc']:.2e}_"
        f"rho_{model_params['rho']}_"
        f"receptors_{model_params['N'] ** 2}_"
        f"_Fmax_{combo.F_max}_{figure_noise_desc(combo)}_"
        f"totalbg_{model_params['total_bg']}_"
        f"n_reps_{config.n_replicates}"
    )
    return {
        "performance": base_dir / f"performance_{core}.svg",
        "psychometric": base_dir / f"psychometric_{core}.svg",
    }


def _metadata(combo: DecoderCombo) -> dict[str, Any]:
    return {
        "train_cue_conc": train_conc_metadata(combo.train_cue_conc),
        "bg_conc": combo.bg_conc,
        "penalty": combo.penalty,
        "rho": combo.rho,
        "fano_factor": combo.fano_factor,
        "noise_desc": csv_noise_desc(combo),
        "F_max": combo.F_max,
    }


def _attach_metadata(df: pd.DataFrame, combo: DecoderCombo, source: str) -> pd.DataFrame:
    out = df.copy()
    for key, value in _metadata(combo).items():
        out[key] = value
    out["source"] = source
    return out


def _read_training_if_present(path: Path, combo: DecoderCombo, source: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return _attach_metadata(pd.read_csv(path), combo, source)


def run_decoder_combo(config: DecoderSweepConfig, combo: DecoderCombo) -> dict[str, Any]:
    paths = result_paths(config, combo)
    fig_paths = figure_paths(config, combo)
    model_params = make_model_params(config, combo)

    if paths["performance"].exists() and not config.force_recompute:
        source = "loaded"
        results_df = pd.read_csv(paths["performance"])
    else:
        source = "computed"
        paths["performance"].parent.mkdir(parents=True, exist_ok=True)
        clf_params = dict(config.clf_params)
        clf_params["penalty"] = combo.penalty
        results_df = run_decoder_experiments_k_datasets(
            config.k,
            config.n_train,
            config.n_test,
            config.cue_concs,
            cue=config.cue,
            train_cue_conc=combo.train_cue_conc,
            n_bg=config.n_bg,
            model_params=model_params,
            clf_params=clf_params,
            penalty=combo.penalty,
            F_max=combo.F_max,
            fano_factor=combo.fano_factor,
            results_dir=config.results_dir,
        )

    if config.make_figures:
        for path in fig_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        plot_performance_curves(
            results_df,
            model_params=model_params,
            savepath=str(fig_paths["performance"]),
            show_plot=config.show_plots,
            noise_desc=figure_noise_desc(combo),
            F_max=combo.F_max,
        )
        plot_psychometric_curve(
            results_df,
            model_params=model_params,
            savepath=str(fig_paths["psychometric"]),
            show_plot=config.show_plots,
            noise_desc=figure_noise_desc(combo),
            F_max=combo.F_max,
        )

    annotated_results = _attach_metadata(results_df, combo, source)
    training_df = _read_training_if_present(paths["training"], combo, source)
    mean_overall_acc = annotated_results.groupby("dataset")["correct"].mean().mean()

    return {
        "combo": combo,
        "source": source,
        "result_df": annotated_results,
        "training_df": training_df,
        "paths": paths,
        "figure_paths": fig_paths,
        "mean_overall_acc": float(mean_overall_acc),
    }


def run_decoder_sweep(config: DecoderSweepConfig) -> DecoderSweepResult:
    combos = iter_combos(config)
    if not combos:
        raise ValueError(f"{config.label}: no parameter combinations to run.")

    if len(combos) == 1:
        summaries = [run_decoder_combo(config, combos[0])]
    else:
        n_jobs = min(config.n_jobs, len(combos))
        summaries = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(run_decoder_combo)(config, combo) for combo in combos
        )

    result_df = pd.concat([item["result_df"] for item in summaries], ignore_index=True)
    training_parts = [item["training_df"] for item in summaries if item["training_df"] is not None]
    training_df = pd.concat(training_parts, ignore_index=True) if training_parts else None

    print(f"{config.label}: {len(combos)} combos")
    for item in summaries:
        combo = item["combo"]
        print(
            f"[{item['source']}] F_max={combo.F_max}, fano_factor={combo.fano_factor}, "
            f"rho={combo.rho}, bg={combo.bg_conc:.2e}, "
            f"train_cue={train_conc_tag(combo.train_cue_conc)}, "
            f"mean acc={item['mean_overall_acc']:.3f}"
        )

    return DecoderSweepResult(
        config=config,
        summaries=summaries,
        result_df=result_df,
        training_df=training_df,
    )


def export_mean_by_complexity(result_df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    out = (
        result_df
        .groupby(["target_conc", "n_bg"])["correct"]
        .agg(["mean"])
        .reset_index()
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def matched_train_test_df(result_df: pd.DataFrame) -> pd.DataFrame:
    scalar = result_df[result_df["train_cue_conc"] != "allconcs"].copy()
    if scalar.empty:
        return scalar
    return scalar[np.isclose(scalar["target_conc"], scalar["train_cue_conc"].astype(float))]


def plot_matched_train_test(
    sweep_result: DecoderSweepResult | pd.DataFrame,
    save_figures: bool = True,
    show_plot: bool = True,
) -> pd.DataFrame:
    if isinstance(sweep_result, DecoderSweepResult):
        df = sweep_result.result_df
        config = sweep_result.config
    else:
        df = sweep_result
        config = DecoderSweepConfig(label="matched_train_test")

    matched_df = matched_train_test_df(df)
    if matched_df.empty:
        raise ValueError("No scalar train/test concentration matches found.")
    if "F_max" not in matched_df.columns:
        matched_df = matched_df.copy()
        matched_df["F_max"] = 10.0

    for (fano_factor, bg_conc, rho, penalty, F_max), subset in matched_df.groupby(
        ["fano_factor", "bg_conc", "rho", "penalty", "F_max"]
    ):
        if save_figures:
            matched_path = (
                config.figures_dir / "matched_conc" /
                f"matched_conc_{penalty}_rho_{rho}_"
                f"bgconc_{bg_conc:.2e}_Fmax_{F_max:g}_noise_fano_{fano_factor:.2e}_n_reps_{config.k}.svg"
            )
            psych_path = (
                config.figures_dir / "manuscript" /
                f"psychometric_cont_learning_{penalty}_rho_{rho}_"
                f"bgconc_{bg_conc:.2e}_Fmax_{F_max:g}_noise_fano_{fano_factor:.2e}_n_reps_{config.k}.svg"
            )
        else:
            matched_path = None
            psych_path = None

        plot_matched_conc_performance_curves(
            subset,
            model_params=None,
            savepath=str(matched_path) if matched_path is not None else None,
            show_plot=show_plot,
        )
        plot_psychometric_curve(
            subset,
            model_params=None,
            savepath=str(psych_path) if psych_path is not None else None,
            show_plot=show_plot,
        )

    return matched_df


def plot_psychometric_comparison(
    matched_df: pd.DataFrame,
    fixed_train_df: pd.DataFrame,
    savepath: str | Path,
    show_plot: bool = True,
) -> None:
    matched_psych = psychometric_summary(matched_df)
    fixed_train_psych = psychometric_summary(fixed_train_df)

    plt.figure(figsize=(5, 5), dpi=600)
    sns.set_context("paper", font_scale=1.7)
    plt.plot(
        matched_psych["target_conc"],
        matched_psych["mean"],
        ".-",
        markersize=10,
        label="Continual learning",
        c="#9d7dd0",
    )
    plt.fill_between(
        matched_psych["target_conc"],
        matched_psych["mean"] - matched_psych["sem"],
        matched_psych["mean"] + matched_psych["sem"],
        alpha=0.25,
        color="#9d7dd0",
    )
    plt.plot(
        fixed_train_psych["target_conc"],
        fixed_train_psych["mean"],
        ".-",
        markersize=10,
        label="Target probe",
        c="#5e78dd",
    )
    plt.fill_between(
        fixed_train_psych["target_conc"],
        fixed_train_psych["mean"] - fixed_train_psych["sem"],
        fixed_train_psych["mean"] + fixed_train_psych["sem"],
        alpha=0.25,
        color="#5e78dd",
    )
    plt.xscale("log")
    plt.xlabel("Target concentration")
    plt.ylabel("Average performance (%)")
    plt.yticks([0.5, 0.75, 1.0], labels=[50, 75, 100])
    sns.despine()
    plt.tight_layout()

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, dpi=600)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_fano_slope_heatmap(
    sweep_result: DecoderSweepResult | pd.DataFrame,
    savepath: str | Path | None = None,
    target_concs: list[float] | None = None,
    fano_factors: list[float] | None = None,
    show_plot: bool = True,
) -> pd.DataFrame:
    if isinstance(sweep_result, DecoderSweepResult):
        df = sweep_result.result_df
        config = sweep_result.config
    else:
        df = sweep_result
        config = None

    if target_concs is not None:
        mask = np.zeros(len(df), dtype=bool)
        for target_conc in target_concs:
            mask |= np.isclose(df["target_conc"], target_conc)
        df = df[mask].copy()
    if fano_factors is not None:
        mask = np.zeros(len(df), dtype=bool)
        for fano_factor in fano_factors:
            mask |= np.isclose(df["fano_factor"], fano_factor)
        df = df[mask].copy()
    if df.empty:
        raise ValueError("No decoding results remain after heatmap filters.")

    avg_df = (
        df
        .groupby(["train_cue_conc", "target_conc", "bg_conc", "fano_factor", "n_bg"])["correct"]
        .mean()
        .reset_index(name="mean_correct")
    )
    slope_rows = []
    for keys, group in avg_df.groupby(["train_cue_conc", "target_conc", "bg_conc", "fano_factor"]):
        train_cue_conc, target_conc, bg_conc, fano_factor = keys
        if group["n_bg"].nunique() < 2:
            continue
        slope_rows.append({
            "train_cue_conc": train_cue_conc,
            "target_conc": target_conc,
            "bg_conc": bg_conc,
            "fano_factor": fano_factor,
            "slope": np.polyfit(group["n_bg"], group["mean_correct"], 1)[0],
        })
    slope_df = pd.DataFrame(slope_rows)
    if slope_df.empty:
        raise ValueError("Cannot compute heatmap slopes; each group has fewer than two n_bg values.")
    avg_slope_df = (
        slope_df
        .groupby(["train_cue_conc", "bg_conc", "fano_factor"])["slope"]
        .mean()
        .reset_index()
    )

    train_cue_conc = avg_slope_df["train_cue_conc"].iloc[0]
    sub_df = avg_slope_df[avg_slope_df["train_cue_conc"] == train_cue_conc]
    pivot_df = sub_df.pivot(index="fano_factor", columns="bg_conc", values="slope")
    if fano_factors is not None:
        pivot_df = pivot_df.reindex(sorted(fano_factors))
    elif config is not None:
        pivot_df = pivot_df.reindex(sorted(config.fano_factors))
    else:
        pivot_df = pivot_df.sort_index()

    plt.figure(figsize=(5, 4), dpi=500)
    ax = sns.heatmap(
        pivot_df,
        cbar_kws={"label": "Background Complexity\nDecodability Slope"},
        vmax=0,
        cmap="rocket",
    )
    ax.invert_yaxis()
    ax.tick_params(axis="both", direction="out")
    ax.minorticks_off()
    ax.set_xticklabels([f"{v:.0e}" for v in pivot_df.columns], rotation=90)
    ax.set_yticklabels([f"{v:.1f}" for v in pivot_df.index], rotation=0)
    plt.ylabel("Fano factor")
    plt.xlabel("Background Concentration")

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", dpi=600)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return avg_slope_df
