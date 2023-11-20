# About

`flexcv` was developed at Hochschule Düsseldorf, Germany by Fabian Rosenthal, Patrick Blättermann and Siegbert Versümer. It is used for the machine learning evaluations in Versümer et al. (2023).

`flexcv` is a Python package and provides a range of features for comparing machine learning models on different datasets with different sets of predictors customizing just about everything around cross validations. It supports both fixed effects and random effects including random slopes.

## Project History

`flexcv` has been developed iteratively to switch and compare methods in calculations for the Soundscape research paper Versümer et al. (2023) which is currently being finished. The paper uses a large dataset of Indoor Soundscape observations, i.e. incorporating both situational and personal features as well as acoustic measurements of the Soundscapes. If you are interested in our research, you can find our previous papers here.

In addition, an interactive visualisation app was [released](https://github.com/radlfabs/DS_Data_Visualization_2023_Fabian_Rosenthal) for the same dataset, which can be adapted to personal needs and to work with other datasets. Have a look at the [technical report](http://dx.doi.org/10.13140/RG.2.2.25156.19845) or the [repo](https://github.com/radlfabs/soundscape-app), if you're interested in the capabilities.

## Creators

__Fabian Rosenthal__ developed the overall structure of the package, designed the interface, implemented the core functionality and modules and is author of the documentation. As a novel addition to the methods for cross validation splits, he contributed a stratified split method for continuous target variables to ensure target distribution across folds. Fabian is no longer affiliated with Hochschule Düsseldorf and maintains the package as a personal project.

__Patrick Blättermann__ implemented the evaluation based on random slopes for the linear mixed effects model and wrote the inner cross validation loop in order to allow customization of the objective function. Patrick served as a consultant on any statistical and machine learning related questions arising.

__Siegbert Versümer__ led the project as product owner in terms of its main ideas and requirements. He contributed a custom objective function (i.e. minimizing the MSE of the inner test set of a nested cross-validation while maintaining a small positive difference between the inner test and training MSE to avoid over- or under-fitting during hyperparameter tuning).
