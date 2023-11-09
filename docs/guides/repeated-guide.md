## Repeated Cross Validation

Some of the cross validation splits are performed with shuffling the data before dividing in train and test splits. Therefore, you might wonder if your evaluation varies for multiple runs.

In the standard configuration, you would seed every run to make it absolutely reproducible. Now we want to explore, how different seeds influence the cross validation results. This is call repeated cross validation. We can still seed this process though by randomly generating a number of seeds. This makes even the repeated CV reproducible.

First, we create our random data set and a basic model mapping just as in a single run.

Second, we instantiate a `RepeatedCV` object. This class not only has the `set`-methods just as CrossValidation but also implements `set_n_repeats()` and `set_neptune()`. We can chain these methods because they also return the class `self` and we use them to set the number of repetitions as well as passing the credentials for Neptune runs. `RepeatedCV` then takes care of instantiating the desired number of runs and logs every single cross validation to it's own neptune run.

Most importantly `RepeatedCV` implements the iteration over single cross validation runs in it's `perform()` method. We can chain `perform()` in the same manner as we are now used to. The last element of our chain should also be `get_results`. This will allow us to inspect summary statistics as a measure of variance in the runs.

Here is the full code to perform cross validation 3 times and get summary statistics for all folds and models.

```python
from flexcv.synthesizer import generate_regression
from flexcv.models import LinearModel
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.repeated import RepeatedCV

# make sample data
X, y, group, random_slopes =generate_regression(10,100,n_slopes=1,noise_level=9.1e-2, random_seed=42

# create a basic model mapping
model_map =ModelMappingDict(
    {
        "LinearModel": ModelConfigDict(
        {
            "model": LinearModel,
        }
    ),
    }
)

credentials = {}

rcv = (
    RepeatedCV()
    .set_data(X, y, group,dataset_name="ExampleData")
    .set_models(model_map)
    .set_n_repeats(3)
    .set_neptune(credentials)
    .perform()
    .get_results()
)

rcv.summary.to_excel("repeated_cv.xlsx")  # save dataframe to excel file
```
