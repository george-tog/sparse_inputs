# coding_model_fanofactor.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import os
import errno
import tempfile
import time

from odor_mix_code.paths import RESULTS_DIR


def _safe_to_csv(df, path, index=False, max_retries=5, base_sleep_s=0.2):
    """
    Robust CSV writer for parallel workers on synced/network-backed folders.
    Writes to a temp file in the target directory and atomically replaces.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for attempt in range(max_retries):
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_csv_", suffix=".csv", dir=out_dir or ".")
        os.close(tmp_fd)
        try:
            df.to_csv(tmp_path, index=index)
            os.replace(tmp_path, path)
            return
        except OSError as err:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            retryable = err.errno in {errno.ECANCELED, errno.EBUSY, errno.EINTR}
            if (not retryable) or (attempt == max_retries - 1):
                raise
            time.sleep(base_sleep_s * (attempt + 1))


def _build_figure_savepath(savepath, F_max=None, noise_desc=None, model_params=None):
    """
    Build a deterministic figure save path and avoid duplicate suffix tokens.
    """
    if F_max is None and model_params is not None:
        F_max = model_params.get("F_max", None)

    base, ext = os.path.splitext(savepath)
    tokens = []
    if F_max is not None:
        fmax_token = f"Fmax_{F_max}"
        if fmax_token not in base:
            tokens.append(fmax_token)
    if noise_desc is not None and noise_desc not in base:
        tokens.append(noise_desc)

    if tokens:
        return f"{base}_{'_'.join(tokens)}{ext}"
    return savepath


def _safe_savefig(path, max_retries=5, base_sleep_s=0.2, **savefig_kwargs):
    """
    Robust figure writer for parallel workers on synced/network-backed folders.
    Saves to a temp file in target directory and atomically replaces.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    suffix = os.path.splitext(path)[1] or ".tmp"
    for attempt in range(max_retries):
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_fig_", suffix=suffix, dir=out_dir or ".")
        os.close(tmp_fd)
        try:
            plt.savefig(tmp_path, **savefig_kwargs)
            os.replace(tmp_path, path)
            return
        except OSError as err:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            retryable = err.errno in {errno.ECANCELED, errno.EBUSY, errno.EINTR}
            if (not retryable) or (attempt == max_retries - 1):
                raise
            time.sleep(base_sleep_s * (attempt + 1))

def init_model_parameters(total_bg=16, N=20, rho=1.0, bg_conc=1e-2, F_max=1, n=4, fano_factor=1.0, train_cue_conc=None, seed=42):
    """
    Initialize model parameters.
    """
    if train_cue_conc is None:
        train_cue_conc = bg_conc
    b = total_bg + 2  # Two cue channels + background odors
    rng = np.random.default_rng(seed=seed)
    lnkappa = rng.standard_normal((b, N, N))
    lneta = rng.standard_normal((b, N, N))
    lneta = rho * lnkappa + np.sqrt(1 - rho**2) * lneta
    lnkappa = -4 * lnkappa
    kappa = np.exp(lnkappa)
    eta = np.exp(lneta)
    return {
        "total_bg": total_bg,
        "N": N,
        "bg_conc": bg_conc,
        "F_max": F_max,
        "n": n,
        "rho": rho,
        "b": b,
        "seed": seed,
        "train_cue_conc": train_cue_conc,
        "lnkappa": lnkappa,
        "lneta": lneta,
        "kappa": kappa,
        "eta": eta,
        "fano_factor": fano_factor
    }

def generate_odor_vecs_vectorized(samples, cue_concs, bg_conc, cue=-1, n_bg=-1, total_bg=16):
    """
    Generate odor vectors in a vectorized manner.
    """
    X = np.zeros((samples, total_bg + 2))
    y = np.empty(samples, dtype=int)
    if np.isscalar(cue_concs):
        cue_concs = np.full(samples, cue_concs)
    if cue == -1:
        rand_cue = np.random.randint(0, 2, size=samples)
    else:
        rand_cue = np.full(samples, cue, dtype=int)
    y = rand_cue.copy()
    X[np.arange(samples), rand_cue] = cue_concs
    if n_bg == -1:
        rand_bg = np.random.randint(0, total_bg + 1, size=samples)
    else:
        rand_bg = np.full(samples, n_bg)
    R = np.random.rand(samples, total_bg)
    ranks = np.argsort(np.argsort(R, axis=1), axis=1)
    mask = ranks < rand_bg[:, None]
    bg_section = np.where(mask, bg_conc, 0.0)
    X[:, 2:] = bg_section
    return X, y

def tensorize_concs(C, n_samp, b, N):
    """
    Broadcast each odor vector into an (N x N) grid.
    """
    return C.reshape(n_samp, b, 1, 1) * np.ones((n_samp, b, N, N))

def compute_kappa_mix_vec(concs, kappa, inv_kappa):
    """
    Compute the effective kappa for a batch of samples.
    """
    C = np.sum(concs, axis=1, keepdims=True)
    betas = concs / C
    inv_kappa_expanded = inv_kappa[None, :, :, :]
    kappa_mix_inv = np.sum(betas * inv_kappa_expanded, axis=1)
    return 1 / kappa_mix_inv

def compute_eta_mix_vec(concs, kappa, eta, inv_kappa):
    """
    Compute the effective eta for a batch of samples.
    """
    C = np.sum(concs, axis=1, keepdims=True)
    betas = concs / C
    k_mix = compute_kappa_mix_vec(concs, kappa, inv_kappa=inv_kappa)
    inv_kappa_expanded = inv_kappa[None, :, :, :]
    eta_expanded = eta[None, :, :, :]
    eta_mix = k_mix * np.sum(eta_expanded * betas * inv_kappa_expanded, axis=1)
    return eta_mix

def compute_activity_vec(concs, kappa, eta, F_max, n, inv_kappa, fano_factor=1.0):
    """
    Compute glomerular activity responses for a batch of samples.
    Add heteroscedastic Gaussian noise with variance proportional to the
    pre-noise mean response.
    """
    k_mix = compute_kappa_mix_vec(concs, kappa, inv_kappa=inv_kappa)
    eta_mix = compute_eta_mix_vec(concs, kappa, eta, inv_kappa=inv_kappa)
    C = np.sum(concs, axis=1)
    activity = F_max / (1 + ((1 + C / k_mix) / (eta_mix * C / k_mix)) ** n)
    std = np.sqrt(fano_factor * np.clip(activity, 0.0, None))
    activity += np.random.normal(0.0, std, size=activity.shape)
    return activity


def generate_datasets(n_train, n_test, cue_concs, train_cue_conc, model_params,
                      cue=-1, n_bg=-1, fano_factor=1.0):
    """
    Generate training and test datasets.
    """
    total_bg = model_params["total_bg"]
    bg_conc = model_params["bg_conc"]
    N = model_params["N"]
    b = model_params["b"] # total odor channels
    F_max = model_params["F_max"]
    n = model_params["n"]
    kappa = model_params["kappa"]
    eta = model_params["eta"]
    inv_kappa = 1.0 / kappa

    # Dispatch based on whether train_cue_conc is a scalar (single conc) or array-like (multiple concs)
    # Robust dispatch: any array-like with length > 1 ⇒ multi-concentration training
    is_arraylike = hasattr(train_cue_conc, "__len__") and not isinstance(train_cue_conc, (str, bytes))
    if is_arraylike and len(train_cue_conc) > 1:
        train_cue_concs = np.asarray(train_cue_conc).reshape(-1)
        L = train_cue_concs.shape[0]
        total_train_samples = n_train * L
        train_cue_concs_vector = np.repeat(train_cue_concs, n_train)
        assert train_cue_concs_vector.shape[0] == total_train_samples, \
            "Internal error: train_cue_concs_vector length mismatch."
        X_train_odor, y_train = generate_odor_vecs_vectorized(
            total_train_samples, train_cue_concs_vector, bg_conc, cue=cue, n_bg=n_bg, total_bg=total_bg
        )
        train_conc_tensor = tensorize_concs(X_train_odor, total_train_samples, b, N)
        train_responses = compute_activity_vec(
            train_conc_tensor, kappa, eta, F_max, n, inv_kappa=inv_kappa, fano_factor=fano_factor
        )
        X_train = train_responses.reshape(L, n_train, -1)
        y_train = y_train.reshape(L, n_train)
    else:
        # single training concentration (scalar or length-1 array-like)
        conc_value = float(np.asarray(train_cue_conc).reshape(())) if is_arraylike else train_cue_conc
        X_train_odor, y_train = generate_odor_vecs_vectorized(
            n_train, conc_value, bg_conc, cue=cue, n_bg=n_bg, total_bg=total_bg
        )
        X_train_tensor = tensorize_concs(X_train_odor, n_train, b, N)
        train_responses = compute_activity_vec(
            X_train_tensor, kappa, eta, F_max, n, inv_kappa=inv_kappa, fano_factor=fano_factor
        )
        X_train = train_responses.reshape(n_train, -1)


    # Batch generate test dataset:
    cue_concs = np.array(cue_concs)
    L = len(cue_concs)
    total_test_samples = L * n_test
    test_cue_concs_vector = np.repeat(cue_concs, n_test)
    X_test_odor, y_test = generate_odor_vecs_vectorized(total_test_samples, test_cue_concs_vector, bg_conc, cue=cue, n_bg=n_bg, total_bg=total_bg)
    test_conc_tensor = tensorize_concs(X_test_odor, total_test_samples, b, N)
    test_responses = compute_activity_vec(test_conc_tensor, kappa, eta, F_max, n, inv_kappa=inv_kappa, fano_factor=fano_factor)
    X_tests = test_responses.reshape(L, n_test, -1)
    y_tests = y_test.reshape(L, n_test)
    all_test_concs = X_test_odor.reshape(L, n_test, -1)
    all_target_concs = test_cue_concs_vector.reshape(L, n_test)

    return X_train, y_train, X_tests, y_tests, all_test_concs, all_target_concs


def run_decoder_experiments_k_datasets(k, n_train, n_test, cue_concs, cue, train_cue_conc, n_bg, model_params, clf_params, penalty, F_max, fano_factor=1.0, results_dir=RESULTS_DIR):
    """
    Generate k datasets and run a decoder experiment on each. Saves performance and weight files.
    """
    # Keep a local copy for filenames/plot metadata and avoid mutating caller state.
    model_params = dict(model_params)
    model_params['fano_factor'] = fano_factor
    model_params['F_max'] = F_max
    results = []
    training_performance = []
    weights_list = []
    for dataset in range(k):
        X_train, y_train, X_tests, y_tests, all_test_concs, all_target_concs = generate_datasets(
            n_train=n_train,
            n_test=n_test,
            cue_concs=cue_concs,
            train_cue_conc=train_cue_conc,
            model_params=model_params,
            cue=cue,
            n_bg=n_bg,
            fano_factor=fano_factor
        )

        if not np.isscalar(train_cue_conc):  # training at multiple target concentrations
            L, n_train, features = X_train.shape
            y_train = y_train.flatten()
            X_train = X_train.reshape(-1, features)
            # print(f'X_train.shape: {X_train.shape}')

        L, n_test_, features = X_tests.shape
        X_tests_flat = X_tests.reshape(-1, features)
        y_tests_flat = y_tests.flatten()
        total_bg = all_test_concs.shape[-1] - 2
        all_test_concs_flat = all_test_concs.reshape(-1, all_test_concs.shape[-1])
        n_bg_vals = (all_test_concs_flat[:, 2:] > 0).sum(axis=1)
        target_concs_flat = all_target_concs.flatten()
        clf = LogisticRegression(random_state=dataset, **clf_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_tests_flat)
        correct = (y_pred == y_tests_flat).astype(float)
        y_train_pred = clf.predict(X_train)
        train_correct = (y_train_pred == y_train).astype(float)
        train_acc = np.mean(train_correct)
        training_performance.append({
            'dataset': dataset,
            'train_accuracy': train_acc
        })
        weights_list.append({
            'dataset': dataset,
            'coef': clf.coef_.tolist(),
            'intercept': clf.intercept_.tolist()
        })
        df = pd.DataFrame({
            'target_conc': target_concs_flat,
            'n_bg': n_bg_vals,
            'correct': correct,
            'dataset': dataset
        })
        grouped = df.groupby(['target_conc', 'n_bg'], as_index=False)['correct'].mean()
        grouped['dataset'] = dataset
        results.append(grouped)
    results_df = pd.concat(results, ignore_index=True)

    noise_desc = f"noise_fano_{fano_factor:.2e}"
    if np.isscalar(train_cue_conc):
        df_savepath = (
            f"{results_dir}/target_bg_conc_noise_grid/{penalty}/performance_{penalty}_trainconc_{train_cue_conc:.2e}_"
            f"bgconc_{model_params['bg_conc']:.2e}_rho_{model_params['rho']}_receptors_{model_params['N']**2}_"
            f"totalbg_{model_params['total_bg']}_Fmax_{F_max}_{noise_desc}_n_reps_{k}.csv"
        )
    else:
        df_savepath = (
            f"{results_dir}/target_bg_conc_noise_grid/{penalty}/performance_{penalty}_trainconc_allconcs_"
            f"bgconc_{model_params['bg_conc']:.2e}_rho_{model_params['rho']}_receptors_{model_params['N']**2}_"
            f"totalbg_{model_params['total_bg']}_Fmax_{F_max}_{noise_desc}_n_reps_{k}.csv"
        )


    _safe_to_csv(results_df, df_savepath, index=False)

    train_df = pd.DataFrame(training_performance)

    if np.isscalar(train_cue_conc):
        train_savepath = (
            f"{results_dir}/target_bg_conc_noise_grid/{penalty}/training_performance_{penalty}_trainconc_{train_cue_conc:.2e}_"
            f"bgconc_{model_params['bg_conc']:.2e}_rho_{model_params['rho']}_receptors_{model_params['N']**2}_"
            f"totalbg_{model_params['total_bg']}_Fmax_{F_max}_{noise_desc}_n_reps_{k}.csv"
        )
    else:
        train_savepath = (
            f"{results_dir}/target_bg_conc_noise_grid/{penalty}/training_performance_{penalty}_trainconc_allconcs_"
            f"bgconc_{model_params['bg_conc']:.2e}_rho_{model_params['rho']}_receptors_{model_params['N']**2}_"
            f"totalbg_{model_params['total_bg']}_Fmax_{F_max}_{noise_desc}_n_reps_{k}.csv"
        )


    _safe_to_csv(train_df, train_savepath, index=False)

    weights_df = pd.DataFrame(weights_list)
    if np.isscalar(train_cue_conc):
        weights_savepath_csv = (
            f"{results_dir}/target_bg_conc_noise_grid/{penalty}/decoder_{penalty}_weights_trainconc_{train_cue_conc:.2e}_"
            f"bgconc_{model_params['bg_conc']:.2e}_rho_{model_params['rho']}_receptors_{model_params['N']**2}_"
            f"totalbg_{model_params['total_bg']}_Fmax_{F_max}_{noise_desc}_n_reps_{k}.csv"
        )
    else:
        weights_savepath_csv = (
            f"{results_dir}/target_bg_conc_noise_grid/{penalty}/decoder_{penalty}_weights_trainconc_allconcs_"
            f"bgconc_{model_params['bg_conc']:.2e}_rho_{model_params['rho']}_receptors_{model_params['N']**2}_"
            f"totalbg_{model_params['total_bg']}_Fmax_{F_max}_{noise_desc}_n_reps_{k}.csv"
        )


    _safe_to_csv(weights_df, weights_savepath_csv, index=False)

    return results_df

def plot_performance_curves(results_df, model_params=None, savepath=None, show_plot=True, noise_desc=None, F_max=None):
    """
    Plot performance curves (accuracy) versus background complexity for each target cue concentration,
    including a shaded area representing the standard error across decoders (mice).

    The function groups the results (across mice) by target cue concentration and n_bg,
    and then computes the mean accuracy and standard error (SEM) for each group.
    The SEM is computed using Pandas’ built-in sem aggregator, which is defined as:

        SEM = std / sqrt(N)

    where N is the number of mice contributing to that group.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns ['target_conc', 'n_bg', 'correct', 'mouse'].
    model_params : dict, optional
        Dictionary of generative model parameters. If provided, these are added to the plot title.
    savepath : str, optional
        File path to save the figure. If provided, the figure is saved to this location.
    """
    colors = [
        "#9d7dd0",
        "#5e78dd",
        "#86d6d3",
        "#7dcc66",
        "#acd05b",
        "#efd962",
        "#dc9c4f",
        "#bd6f6d",
        "#c37ab4"
    ]

    # Group over all mice for each combination of target_conc and n_bg,
    # and compute both the mean accuracy and its SEM.
    summary = results_df.groupby(['target_conc', 'n_bg'])['correct'].agg(['mean', 'sem']).reset_index()

    plt.figure(figsize=(7, 5), dpi=600)

    # Get unique target cue concentrations and sort them.
    unique_cues = sorted(summary['target_conc'].unique())

    # Ensure there are no more than 16 distinct target cue concentrations.
    if len(unique_cues) > 16:
        raise ValueError("More than 16 unique target cue concentrations provided. Maximum allowed is 16.")

    # Generate distinct colors for each target cue concentration.
    sns.set_context("paper", font_scale=1.7)

    # Plot the performance curve for each cue concentration.
    for i, cue_conc in enumerate(unique_cues):
        sub = summary[summary['target_conc'] == cue_conc].sort_values('n_bg')
        plt.plot(sub['n_bg'], sub['mean'], label=f'{cue_conc:.2e}', color=colors[i])
        plt.fill_between(sub['n_bg'], sub['mean'] - sub['sem'], sub['mean'] + sub['sem'],
                         alpha=0.3, color=colors[i])

    plt.xlabel('Background #')
    plt.ylabel('Decoder Accuracy (%)')

    plt.yticks([0.5, 0.75, 1.0], [50, 75, 100])

    # Add model parameters to the title if provided.
    if model_params is not None:
        title_str = (f"Train Cue Conc: {model_params['train_cue_conc']}, BG Conc: {model_params['bg_conc']}, "
                     f"$\\rho$: {model_params['rho']}, FF: {model_params['fano_factor']}")
    else:
        title_str = "Decoder Performance"
    plt.title(title_str)
    plt.legend(title='Target Odor\nConcentration', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine()
    plt.tight_layout()

    if savepath is not None:
        final_savepath = _build_figure_savepath(
            savepath,
            F_max=F_max,
            noise_desc=noise_desc,
            model_params=model_params
        )
        _safe_savefig(final_savepath, bbox_inches='tight')
        print(f"Figure saved to {final_savepath}")
    else:
        plt.show()

    if not show_plot:
        plt.close()

def psychometric_summary(df):
    # Match your newer plot_psychometric_curve behavior:
    # average within dataset first if dataset labels exist.
    if "dataset" in df.columns:
        dataset_level = (
            df.groupby(["target_conc", "dataset"], as_index=False)["correct"]
            .mean()
        )
        out = (
            dataset_level.groupby("target_conc")["correct"]
            .agg(mean="mean", sem="sem")
            .reset_index()
        )
    else:
        out = (
            df.groupby("target_conc")["correct"]
            .agg(mean="mean", sem="sem")
            .reset_index()
        )

    out["sem"] = out["sem"].fillna(0.0)
    return out.sort_values("target_conc")


def plot_psychometric_curve(results_df, model_params=None, savepath=None, show_plot=True, noise_desc=None, F_max=None):
    """
    Plot a psychometric curve of average performance versus target concentration.

    This function groups the raw performance data (ignoring background complexity)
    by target cue concentration (averaging across all n_bg values and datasets) and plots the
    mean accuracy as a function of target concentration on a log-scaled x-axis, with a
    shaded region showing mean +/- SEM.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns ['target_conc', 'n_bg', 'correct', 'mouse'].
    model_params : dict, optional
        Dictionary of generative model parameters. If provided, these are added to the plot title.
    savepath : str, optional
        File path to save the figure. If provided, the figure is saved at this location.
    """
    # Compute psychometric mean/SEM per target concentration.
    # If dataset labels exist, first average across n_bg within each dataset to
    # avoid treating each n_bg point as an independent replicate.
    if 'dataset' in results_df.columns:
        dataset_level = (
            results_df
            .groupby(['target_conc', 'dataset'], as_index=False)['correct']
            .mean()
        )
        psych_df = (
            dataset_level
            .groupby('target_conc', as_index=False)['correct']
            .agg(['mean', 'sem'])
            .reset_index()
            .sort_values('target_conc')
        )
    else:
        psych_df = (
            results_df
            .groupby('target_conc', as_index=False)['correct']
            .agg(['mean', 'sem'])
            .reset_index()
            .sort_values('target_conc')
        )
    psych_df['sem'] = psych_df['sem'].fillna(0.0)

    colors = [
            "#9d7dd0",
            "#5e78dd",
            "#86d6d3",
            "#7dcc66",
            "#acd05b",
            "#efd962",
            "#dc9c4f",
            "#bd6f6d",
            "#c37ab4"
        ]
    plt.figure(figsize=(4, 4), dpi=600)
    sns.set_context("paper", font_scale=1.7)

    plt.plot(psych_df['target_conc'], psych_df['mean'], '.-', markersize=10, c=colors[0])
    plt.fill_between(
        psych_df['target_conc'],
        psych_df['mean'] - psych_df['sem'],
        psych_df['mean'] + psych_df['sem'],
        color=colors[0],
        alpha=0.25
    )
    plt.xscale('log')
    plt.xlabel("Target Concentration")
    plt.ylabel("Average Performance (%)")
    plt.yticks([0.5, 0.75, 1.0], labels=[50, 75, 100])
    if model_params is not None:
        title_str = (f"Train Cue Conc: {model_params['train_cue_conc']}, BG Conc: {model_params['bg_conc']}, "
                     f"$\\rho$: {model_params['rho']}")
    else:
        title_str = "Psychometric Curve"
    plt.title(title_str, pad=10)
    sns.despine()
    # plt.tight_layout()

    if savepath is not None:
        final_savepath = _build_figure_savepath(
            savepath,
            F_max=F_max,
            noise_desc=noise_desc,
            model_params=model_params
        )
        _safe_savefig(final_savepath, bbox_inches='tight')
        print(f"Psychometric curve saved to {final_savepath}")
    else:
        plt.show()

    if not show_plot:
        plt.close()

def plot_matched_conc_performance_curves(results_df, model_params=None, savepath=None, show_plot=True, noise_desc=None, F_max=None):
    '''
    results_df: dataframe where training and testing concentrations are matched
    '''
    summary = results_df.groupby(['target_conc', 'n_bg'])['correct'].agg(['mean', 'sem']).reset_index()
    plt.figure(figsize=(7,5), dpi=600)
    sns.set_context("paper", font_scale=1.7)

    summary = results_df.groupby(['target_conc', 'n_bg'])['correct'].agg(['mean', 'sem']).reset_index()

    # Get unique target cue concentrations and sort them.
    unique_cues = sorted(summary['target_conc'].unique())

    # Ensure there are no more than 16 distinct target cue concentrations.
    if len(unique_cues) > 16:
        raise ValueError("More than 16 unique target cue concentrations provided. Maximum allowed is 16.")

    colors = [
        "#9d7dd0",
        "#5e78dd",
        "#86d6d3",
        "#7dcc66",
        "#acd05b",
        "#efd962",
        "#dc9c4f",
        "#bd6f6d",
        "#c37ab4"
    ]

    # Get unique target cue concentrations and sort them.
    unique_cues = sorted(summary['target_conc'].unique())

    # Ensure there are no more than 16 distinct target cue concentrations.
    if len(unique_cues) > 16:
        raise ValueError("More than 16 unique target cue concentrations provided. Maximum allowed is 16.")

    # Generate distinct colors for each target cue concentration.
    line_styles = ['-'] #, '--', '-.', ':'] * int(16 / 2)

    # Plot the performance curve for each cue concentration.
    for i, cue_conc in enumerate(unique_cues):
        sub = summary[summary['target_conc'] == cue_conc].sort_values('n_bg')
        plt.plot(sub['n_bg'], sub['mean'], label=f'{cue_conc:.2e}', color=colors[i], linestyle=line_styles[0])
        plt.fill_between(sub['n_bg'], sub['mean'] - sub['sem'], sub['mean'] + sub['sem'],
                            alpha=0.3, color=colors[i])

    plt.xlabel('Background #')    
    plt.ylabel('Decoder Accuracy (%)')
    plt.yticks(ticks=[0.5, 0.75, 1.00], labels=[50, 75, 100])

    # Add model parameters to the title if provided.
    if model_params is not None:
        title_str = (f"Train Cue Conc: {model_params['train_cue_conc']}, BG Conc: {model_params['bg_conc']}, "
                    f"$\\rho$: {model_params['rho']}, receptors: {model_params['N']**2}")
    else:
        title_str = "Decoder Performance"
    plt.title("Matched train-test target conc")
    plt.legend(title='Target Odor\nConcentration', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine()
    plt.tight_layout()

    if savepath is not None:
        final_savepath = _build_figure_savepath(
            savepath,
            F_max=F_max,
            noise_desc=noise_desc,
            model_params=model_params
        )
        _safe_savefig(final_savepath, bbox_inches='tight', dpi=600)
        print(f"Figure saved to {final_savepath}")
    else:
        plt.show()

    if not show_plot:
        plt.close()

def population_sparsity(resp):
    """
    resp: np.array with shape (glomeruli, odors)
    returns: np.array with shape (odors,)
    """
    N_glom = resp.shape[0]
    prefactor = 1 / (1 - 1 / N_glom)
    numerator = np.mean(resp, axis=0) ** 2
    denominator = np.mean(resp ** 2, axis=0)
    return prefactor * (1 - numerator / denominator)
