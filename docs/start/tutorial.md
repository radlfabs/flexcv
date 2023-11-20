To get started with `flexcv`, we take you through a couple of quick and basic code examples. You will learn how to set up `flexcv` with a linear model and how the interaction with the interface class works in practical use. At the end of this section you will be familiar with the basic concepts of `flexcv` and will be able to use it for your own projects.

###### Linear Model

First we will use a LinearModel on a randomly generated regression dataset. Because Linear Models do not have any hyperparameters, we naturally don't need an inner cross validation loop.

```py
# import the class interface, data generator and model
from flexcv import CrossValidation
from flexcv.synthesizer import generate_regression
from flexcv.models import LinearModel
  
# make sample data
X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42)

# instantiate our cross validation class
cv = CrossValidation()

# now we can use method chaining to set up our configuration perform the cross validation
results = (
    cv
    .set_data(X, y, group, dataset_name="ExampleData")
    .set_splits(method_outer_split="GroupKFold", method_inner_split="KFold")
    .add_model(LinearModel)
    .set_splits(break_cross_val=True)
    .perform()
    .get_results()
)

# results has a summary property which returns a dataframe
# we can simply call the pandas method "to_excel"
results.summary.to_excel("my_cv_results.xlsx")

```
