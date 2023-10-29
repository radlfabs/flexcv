import logging
from pprint import pformat

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import rpy2.robjects as ro
import shap
from neptune.types import File

from . import plot
from .plot import permutation_importance
from .statsmodels_diagnostics import LinearRegDiagnostic

mpl.use("Agg")
logger = logging.getLogger(__name__)


def lm_post(results_all_folds, fold_result, run, *args, **kwargs):
    # LM is the only one where the regular logging wont work with get_params()
    # therefore, we need to get the summary via the model object and simply overwrite the empty logs
    params = fold_result.fit_result.get_summary()

    run[f"{fold_result.model_name}/Summary/{fold_result.k}"].upload(
        File.from_content(params, extension="html")
    )
    results_all_folds[fold_result.model_name]["parameters"][fold_result.k] = params

    vif, fig, ax = LinearRegDiagnostic(fold_result.fit_result.md_)()  # type: ignore   instance has to be called after __init__
    run[f"{fold_result.model_name}/VIF/{fold_result.k}"].upload(File.as_html(vif))
    run[f"{fold_result.model_name}/Plots"].log(fig)
    del fig
    del vif
    del ax
    return results_all_folds


def lmer_post(results_all_folds, fold_result, run, *args, **kwargs):
    # LM is the only one where the regular logging wont work with get_params()
    # therefore, we need to get the summary via the model object and simply overwrite the empty logs
    params = fold_result.fit_result.get_summary()

    run[f"{fold_result.model_name}/Summary/{fold_result.k}"].upload(
        File.from_content(params, extension="html")
    )
    results_all_folds[fold_result.model_name]["parameters"][fold_result.k] = params
    return results_all_folds


def rf_post(results_all_folds, fold_result, run, *args, **kwargs):
    run[f"{fold_result.model_name}/BestParams/{fold_result.k}"] = pformat(
        fold_result.best_params
    )
    explainer = shap.TreeExplainer(fold_result.best_model)
    shap_values = explainer.shap_values(fold_result.X_train)
    shap_explainer = explainer(fold_result.X_train)
    plot.plot_shap(
        shap_values=shap_values,
        X=fold_result.X_train,
        run=run,
        log_destination=f"{fold_result.model_name}/SHAP/Fold",
        dependency=False,
    )
    if "shap_values" not in results_all_folds[fold_result.model_name].keys():
        results_all_folds[fold_result.model_name]["shap_values"] = [shap_values]
    else:
        results_all_folds[fold_result.model_name]["shap_values"].append(shap_values)

    shap_values_test = explainer.shap_values(fold_result.X_test)
    plot.plot_shap(
        shap_values=shap_values_test,
        X=fold_result.X_test,
        run=run,
        log_destination=f"{fold_result.model_name}/SHAP/Test_Fold",
        dependency=False,
    )

    return results_all_folds


def xgboost_post(results_all_folds, fold_result, run, *args, **kwargs):
    explainer = shap.TreeExplainer(fold_result.best_model)
    shap_values = explainer.shap_values(fold_result.X_train)
    shap_explainer = explainer(fold_result.X_train)
    plot.plot_shap(
        shap_values=shap_values,
        X=fold_result.X_train,
        run=run,
        log_destination=f"{fold_result.model_name}/SHAP/Train_Fold",
        dependency=False,
    )
    if "shap_values" not in results_all_folds[fold_result.model_name].keys():
        results_all_folds[fold_result.model_name]["shap_values"] = [shap_values]
    else:
        results_all_folds[fold_result.model_name]["shap_values"].append(shap_values)

    run[f"{fold_result.model_name}/BestParams/{fold_result.k}"] = pformat(
        fold_result.best_params
    )

    shap_values_test = explainer.shap_values(fold_result.X_test)
    plot.plot_shap(
        shap_values=shap_values_test,
        X=fold_result.X_test,
        run=run,
        log_destination=f"{fold_result.model_name}/SHAP/Test_Fold",
        dependency=False,
    )

    return results_all_folds


def mars_post(results_all_folds, fold_result, run, *args, **kwargs):
    with plt.style.context("ggplot"):
        # try:
        #     Path("tmp_imgs").mkdir(parents=True, exist_ok=True)

        #     for i in range(1, 5):
        #         run[f"MARS/Plots/{fold_result.k}/{i}"].upload(
        #             File(f"tmp_imgs/mars_plot_{i}.png")
        #         )
        # except Exception as e:
        #     logger.info(e)

        imp_df: pd.DataFrame = fold_result.best_model.get_variable_importance(
            kwargs["features"]
        )
        run["MARS/FeatImportance/Table"].upload(File.as_html(imp_df))
        for col in imp_df.columns:
            # plot all rows of col where col is not 0
            fig = plt.figure()
            tmp = imp_df[col]
            tmp = tmp[tmp != 0]
            try:
                tmp.plot.barh()
                plt.title(f"{col} Variable Importance")
                run[f"MARS/FeatImportance/{fold_result.k}"].log(fig)
            except Exception as e:
                logger.info(f"{e}")
                logger.info("Could not plot MARS barplot. Continuing.")
            del fig
            plt.close()

        run[f"{fold_result.model_name}/BestParams/{fold_result.k}"] = pformat(
            fold_result.best_params
        )

    return results_all_folds


def svr_post(results_all_folds, fold_result, run, *args, **kwargs):
    with plt.style.context("ggplot"):
        run[f"{fold_result.model_name}/BestParams/{fold_result.k}"] = pformat(
            fold_result.best_params
        )

        fig, tmp_df = permutation_importance(
            fold_result.best_model,
            fold_result.model_name,
            fold_result.X_test,
            fold_result.y_test,
            fold_result.X_test.columns,
        )
        run["SVR/PermFeatImportance/Figures"].log(fig)
        run["SVR/PermFeatImportance/Table"].upload(File.as_html(tmp_df))

    return results_all_folds


def expectation_maximation_post(
    results_all_folds, fold_result, y_pred_base, run, *args, **kwargs
):
    run[f"{fold_result.model_name}/BestParams/{fold_result.k}"] = pformat(
        {
            "max_iterations": fold_result.fit_result.max_iterations,
        }
    )

    plot.plot_merf_results(
        y=fold_result.y_test,
        yhat_base=y_pred_base,
        yhat_me_model=fold_result.y_pred,
        model=fold_result.fit_result,
        model_name=kwargs["mixed_name"],
        max_iterations=fold_result.fit_result.max_iterations,
        run=run,
    )
    plt.close()
    return results_all_folds


if __name__ == "__main__":
    pass
