import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.inspection import \
    permutation_importance as sk_permutation_importance
from xgboost import plot_importance

mpl.use("Agg")


def matplotlib_settings() -> int:
    mpl.rcParams["figure.figsize"] = (11, 8)
    plt.rc("font", size=8)
    plt.rc("axes", titlesize=8)
    plt.rc("axes", labelsize=8)
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)
    plt.rc("legend", fontsize=8)
    plt.rc("axes", linewidth=1)
    dpi = 100
    return dpi


def set_axes_params(ax) -> None:
    ax.tick_params(axis="both", which="both", length=3)
    return None


def plot_merf_training_stats(run, model, model_name, num_clusters_to_plot=5) -> None:
    """
    * Generalized log-likelihood across iterations
    * trace and determinant of Sigma_b across iterations
    * sigma_e across iterations
    * bi for num_clusters_to_plot across iterations
    * a histogram of the final learned bi

    Args:
        model (MERF): trained MERF model
        num_clusters_to_plot (int): number of example bi's to plot across iterations
        meta_string (string): A string for use as additional info in filename.

    Returns:
        (matplotlib.pyplot.fig): figure. Also draws to display.
    """
    # get number of columns of model.trained_b

    dpi = matplotlib_settings()
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=dpi)

    # Plot GLL
    axs[0, 0].plot(model.gll_history)
    axs[0, 0].grid("on")
    axs[0, 0].set_ylabel("GLL")
    axs[0, 0].set_title("GLL")
    set_axes_params(axs[0, 0])

    # Plot trace and determinant of Sigma_b (covariance matrix)
    det_sigmaB_history = [np.linalg.det(x) for x in model.D_hat_history]
    trace_sigmaB_history = [np.trace(x) for x in model.D_hat_history]
    axs[0, 1].plot(det_sigmaB_history, label="$det(\sigma_b)$")  # det($sigma_b$)"
    axs[0, 1].plot(
        trace_sigmaB_history, label="$trace(\sigma_b)$"
    )  # "trace($sigma_b$)"
    axs[0, 1].grid("on")
    axs[0, 1].legend()
    axs[0, 1].set_title("Trace and Determinant of $\sigma_b$")
    set_axes_params(axs[0, 1])

    # Plot sigma_e across iterations
    axs[1, 0].plot(model.sigma2_hat_history)
    axs[1, 0].grid("on")
    axs[1, 0].set_ylabel("$\hat\sigma_e$")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_title("$\hat\sigma_e$ vs iterations")
    set_axes_params(axs[1, 0])

    # Plot bi across iterations
    b_hat_history_df = model.get_bhat_history_df()
    for cluster_id in model.cluster_counts.index[0:num_clusters_to_plot]:
        axs[1, 1].plot(
            b_hat_history_df.xs(cluster_id, level="cluster"), label=cluster_id
        )
    axs[1, 1].grid("on")
    axs[1, 1].set_ylabel("$b_hat$")
    axs[1, 1].set_xlabel("Iteration")
    tmp_title = "$b_i$ vs iterations\n" + f"({num_clusters_to_plot} clusters shown)"
    axs[1, 1].set_title(tmp_title)
    set_axes_params(axs[1, 1])
    run[f"{model_name}/Plots/Training_Stats"].log(fig)
    plt.close(fig)

    num_random_effects = model.trained_b.shape[1]
    dpi = matplotlib_settings()
    fig, axs = plt.subplots(nrows=1, ncols=num_random_effects, dpi=dpi)
    model.trained_b.hist(bins=100, ax=axs)
    if num_random_effects == 1:
        axs.set_xlabel("$b_i$")
        axs.set_title("Distribution of $b_i$ for Random Intercepts")
    else:
        try:
            for i in range(num_random_effects):
                axs[i].set_xlabel("$b_i$")
                axs[i].set_title(
                    "Distribution of $b_i$ for Random Effect " + str(i + 1)
                )

        except ValueError:
            print("Plotting Error:")
            print(
                "hist method requires numerical or datetime columns, nothing to plot."
            )
            run[
                "Errors"
            ] = "Error in plot_merf_training_stats: model.trained_b.hist(bins=100, ax=ax)"
    plt.tight_layout()
    run[f"{model_name}/Plots/Training_Hists"].log(fig)
    plt.close(fig)

    return None


def plot_merf_results(
    y,
    yhat_base,
    yhat_me_model,
    model,
    model_name,
    run,
    **kwargs,
) -> None:
    dpi = matplotlib_settings()
    fig = plt.figure(dpi=dpi)
    plt.plot(y, yhat_base, ".", label=f"Base Model (w/o EM))")
    plt.plot(y, yhat_me_model, ".", label=f"{model_name} (+EM)")
    plt.grid("on")
    plt.xlabel("y (actual)")
    plt.ylabel("$\hat y (predicted)$")
    plt.legend()
    plt.tight_layout()

    run[f"{model_name}/Plots/Predictions"].log(fig)
    plt.close(fig)
    plot_merf_training_stats(
        run=run,
        model=model,
        model_name=model_name,
        num_clusters_to_plot=5,
    )
    plt.close(fig)
    return None


def plot_shap(
    shap_values,
    X: pd.DataFrame,
    run=None,
    log_destination="SHAP/",
    dependency: bool = True,
    k_features: pd.Series = None,
):
    """Creates SHAP summary beeswarm and dependency plots (if set to True) and logs them to a Neptune Run."""
    plt.close()
    plt.cla()
    shap.summary_plot(shap_values, X, show=False)
    f = plt.gcf()
    run[f"{log_destination}Importance"].log(f)
    del f

    if dependency:
        for col_name in k_features:
            plt.close()
            plt.cla()
            shap.dependence_plot(col_name, shap_values, X, show=False, alpha=0.5)
            f = plt.gcf()
            run[f"{log_destination}Dependency"].log(f)
            del f

    return None


def plot_qq(
    y: pd.Series,
    yhat: pd.Series,
    run=None,
    model_name: str = "LM",
    log_destination: str = "LM_Plots/QQ/",
):
    """Creates QQ plot and logs it to a Neptune Run."""
    plt.close()
    plt.cla()
    fig = sm.qqplot(y - yhat, line="r")
    plt.title(f"QQ plot of residuals - {model_name}")
    run[f"{log_destination}{model_name}_QQ"].log(fig)
    del fig
    return None


def permutation_importance(
    model, model_name, X, y, features
) -> tuple[plt.Figure, pd.DataFrame]:
    fig = plt.figure()
    perm_importance = sk_permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )
    features = np.array(features)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel(f"{model_name} Permutation Importance")
    df = pd.DataFrame(
        {
            "feature": features[sorted_idx],
            "importance": perm_importance.importances_mean[sorted_idx],
        }
    )
    return fig, df


if __name__ == "__main__":
    pass
