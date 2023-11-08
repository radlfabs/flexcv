# Overview

This user guide will give you a detailled guide on how to use `flexcv` functions and objects.

Let's dive into [how to fit a Random Forest Regressor ](guides/rf-regressor.md)on your data by tuning a hyperparameter in the inner cross validation and evaluate the model's performance in the outer cross validation.

Also, our guide on [how to evaluate random effects](random-effects.md) takes you on the journey through the land of hierarchical data and gives an overview what to consider when facing machine learning problems that have a grouped structure. `flexcv` has some tools for that.

You might wonder, if your processes are influenced by randomness. Our guide on [how to set up a repeated cross validation](repeated.md) tackles this topic and shows how to use the `RepeatedCV` class with `flexcv`.

You still can not decide on the model type? No problem with `flexcv` since it allows to [run multiple models in a single run](multiple-models.md). This guide shows how you set up the configuration for multiple models using our `ModelMappingDict` class. Calling nested cross validation using this syntax is easy as can be.
