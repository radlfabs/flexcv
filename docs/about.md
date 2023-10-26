# About

`flexcv` was developed at Hochschule Düsseldorf, Germany by Fabian Rosenthal, Patrick Blättermann and Siegbert Versümer. It is used for the machine learning evaluations in Versümer et al. (2023).

`flexcv` is a Python package and provides a range of features for comparing machine learning models on different datasets with different sets of predictors customizing just about everything around cross validations. It supports both fixed effects and random effects including random slopes.

__Fabian Rosenthal__ developed the overall structure of the package, designed the interface, and implemented the core functionality, modules and documentation. As a novel addition to the methods for cross validation splits, he contributed a stratified split method for continuous target variables to ensure target distribution across folds.

__Patrick Blättermann__ implemented the evaluation based on random slopes for the linear mixed effects model and wrote the inner cross validation loop in order to allow customization of the objective function. Patrick served as a consultant on any statistical and machine learning related questions arising.

__Siegbert Versümer__ led the project as product owner in terms of its main ideas and requirements. He contributed a custom objective function (i.e. minimizing the MSE of the inner test set of a nested cross-validation while maintaining a small positive difference between the inner test and training MSE to avoid over- or under-fitting during hyperparameter tuning). He further provided the data for the evaluation in Versümer et al. (2023).
