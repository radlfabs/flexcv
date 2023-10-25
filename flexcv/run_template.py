import logging

from main import main
from data_loader import DataLoadConfig
from run import RunConfigurator
import cross_val_split as cross_val_split
from funcs import add_module_handlers

logger = logging.getLogger(__name__)

# make a personal_run.py module with this template
# exclude it from git

if __name__ == "__main__":
    add_module_handlers(logger)

    dataset = "HSDD"  # , "ISD"]
    target = "CFAEventfulness"  # , "CFAPleasantness"]
    model_level = 4
    custom = cross_val_split.CrossValMethod.CUSTOM
    strat = cross_val_split.CrossValMethod.STRAT
    gkf = cross_val_split.CrossValMethod.GROUPKFOLD
    cv_out, cv_in = (custom, gkf)  # , (gkf, gkf))

    main(
        RunConfigurator(
            DataLoadConfiguration=DataLoadConfig(
                dataset_name=dataset,
                model_level=model_level,
                target_name=target,
                slopes="LAeq",  # TODO enable slopes here
            ),
            cross_val_method=cv_out,
            cross_val_method_in=cv_in,
            neptune_on=True,
            scale_in=True,
            scale_out=True,
            em_max_iterations=5,
            diagnostics=False,
            n_splits=5,
            n_trials=3,
            refit=False,
            model_selection="best",
            predict_known_groups_lmm=True,
            break_cross_val=True,
        )
    )
