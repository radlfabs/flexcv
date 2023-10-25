# Getting Started

To get started with `flexcv`, you can create a new Python file and import the necessary modules:

```py
from flexcv import CrossValidation
from flexcv.cv_split import CrossValMethod
import pandas as pd
```

Next, you can create a new instance of the `CrossValidation` class:

```py
cv = CrossValidation()
```

Let's have a look at the class definition:

```py
@dataclass
class CrossValidation(BaseConfigurator):
    """This dataclass is constructed using the following configuration classes:
    - DataConfigurator
    - CrossValConfigurator
    - RunConfigurator
    - OptimizationConfigurator (optional)
    - MixedEffectsConfigurator (optional)
    It also takes an optional random seed. If no seed is passed, it defaults to 42. 
    It is used to pass all the necessary parameters to the cross validation function.

    Methods:
    """
    data_config: DataConfigurator | None = None
    cross_val_config: CrossValConfigurator | None = None
    run_config: RunConfigurator | None = None
    model_mapping: ModelMappingDict | None = None
    optim_config: OptimizationConfigurator | None = None
    mixed_effects_config: MixedEffectsConfigurator | None = None
    random_seed: int = 42
```



## Configuring the Data

Before performing cross-validation, you need to configure the data. You can do this using the `data_config` method:

```py
cv.set_data(
    dataset_name="dataset_name",
    target_name="target_name",
    X=pd.DataFrame(),
    y=pd.Series(),
    group=pd.Series(),
    slopes=pd.Series(),
    model_level="mixed",
)
```

Here, you need to specify the name of the dataset, the name of the target variable, and the input features and target variable as pandas DataFrames and Series, respectively. You can also specify the grouping variable, random slopes, and model level.

## Setting the Data

After configuring the data, you need to set it using the `set_data` method:

```py
cv.set_data(
    dataset_name="dataset_name",
    target_name="target_name",
    X=pd.DataFrame(),
    y=pd.Series(),
    group=pd.Series(),
    slopes=pd.Series(),
    model_level="mixed",
)
```

This method is similar to `data_config`, but it sets the data configuration for the cross-validation instance.

## Configuring Cross-Validation

Next, you need to configure the cross-validation settings. You can do this using the `set_cross_val` method:

```py
cv.set_cross_val(
    cross_val_method=CrossValMethod.KFOLD,
    cross_val_method_in=CrossValMethod.KFOLD,
    n_splits=5,
    scale_in=True,
    scale_out=True,
    break_cross_val=False,
    metrics=None
)
```

## Alternative approach
Instead of instantiating the `CrossValidation` class without any arguments, you can also pass the necessary parameters to the constructor. For example, you can configure the data, cross-validation settings, and run settings before creating a `CrossValidation` instance by importing the Configuration classes separately:

```py
from flexcv import CrossValidation
from flexcv.cv_split import CrossValMethod
from flexcv.interface import DataConfigurator, CrossValConfigurator, RunConfigurator

data_config = DataConfigurator(
    dataset_name="dataset_name",
    target_name="target_name",
    X=pd.DataFrame(),
    y=pd.Series(),
    group=pd.Series(),
    slopes=pd.Series(),
    model_level="mixed",
)

cross_val_config = CrossValConfigurator(
    cross_val_method=CrossValMethod.KFOLD,
    cross_val_method_in=CrossValMethod.KFOLD,
    n_splits=5,
    scale_in=True,
    scale_out=True,
    break_cross_val=False,
    metrics=None
)

run_config = RunConfigurator(

)

cv = CrossValidation(
    data_config=data_config,
    cross_val_config=cross_val_config,
    run_config=run_config
)
```

Here, you can specify the cross-validation method, the number of splits, and whether to scale the input and output data. You can also specify whether to break cross-validation if a certain condition is met, and the metrics to use for cross-validation.

## Performing Cross-Validation

After configuring the data and cross-validation settings, you can perform cross-validation using the `perform` method:

```py
results = cv.perform()
```

This method returns a dictionary-like object of the cross-validation results.

