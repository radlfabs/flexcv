# flexcv v23.0.1-beta

## Fixing Stuff in Beta

And here we go again...
It turned out that my GitHub Actions workflow files was bugged and the workflow did not fail with PyTest discovery errors as it was supposed to.
Therefore, the early fix was absolutely necessary to make it all work.

### Bug Fixes

- Fix: NameError in yaml_parser.py
- Moves a data set geneartor template file to the package for easier testing


# flexcv v23.0-beta

## Initial Beta Release

Welcome to the first release of flexcv! This version marks the beginning of a powerful and versatile library designed to simplify nested cross validation tasks in Python.

### Key Features

- **Feature**: Interface based nested cross validation allows deep customizations without re-implementing the wheel.
- **Mixed Effects**: Integrating MERF to apply correction based on clusters in data.
- **Experiment Tracking**: Neptune integrations for extensive logging
- **Documentation**: Initial documentation outlining the package structure and providing usage examples for various ML tasks.

### What's Included

flexcv 23.0.0 provides:

- Interface class to set up and perform nested cross validation.
- Customizability of cross validation settings.
- Suited for hierarchical/clustered data with mixed effects.
- SOTA hyperparameter optimization with Optuna.
- Extensive online logging thanks to Neptune.ai.
- Custom solutions for stratified kFold on continuous target variables.
- Custom objective scoring to balance out over-/underfitting.
- Yaml parser to read in model configuration.
- Multiple Models per Run.
- Repeated CV.
- Detailled documentation.
- Multiple user guides across different ML tasks.
- Great test coverage.

### Notes

This inaugural release lays the foundation for future enhancements and feature additions. We welcome feedback, contributions, and suggestions as we continue to expand flexcv.

Thank you for being part of this journey!
