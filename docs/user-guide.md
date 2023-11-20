# Overview

This user guide will give you a detailled information on how to use `flexcv` functions and objects.

If you want to learn about nested cross validation in general and how we implemented it as a workflow in `flexcv`, check out our [nested cross validation guide](guides/flow.md). This gives you a step by step overview of the process and explains the motivation behind nested cross validation.

You can learn about the `neptune` integration and tracking your experiments by checking out our [neptune integration guide](guides/neptune-integration.md). You will learn how to use this super cool MLOps tool and leverage it's power to give you great insights in data and model performance.

Let's dive into [how to fit a Random Forest Regressor ](guides/rf-regressor.md)on your data by tuning a hyperparameter in the inner cross validation and evaluate the model's performance in the outer cross validation.

Also, our guide on [how to evaluate random effects](guides/random-effects.md) takes you on the journey through the land of hierarchical data and gives an overview what to consider when facing machine learning problems that have a grouped structure. `flexcv` has some tools for that.

You might wonder, if your processes are influenced by randomness. Our guide on [how to set up a repeated cross validation](guides/repeated-guide.md) tackles this topic and shows how to use the `RepeatedCV` class with `flexcv`.

You still can not decide on the model type? No problem with `flexcv` since it allows to [run multiple models in a single run](guides/multiple-models.md). This guide shows how you set up the configuration for multiple models using different methods on the interface class or by storing it in a yaml file.
