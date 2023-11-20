This guide will help you get started with the `flexcv` package quickly. It will show you how to install the package and how to use it to compare machine learning models on your data. At the end of this section you will have a working environment to start using `flexcv` for your own projects.

## Installation

You can just install the package from PyPI using `pip`:

```bash
pip install flexcv
```

We support Python 3.10 and 3.11 at the moment. We will support 3.12 as soon as some dependencies are updated as well.

##### Using venv

To separate Python environments on your system, we recommend to use some kind of virtual environment. One way is to use the `venv` package from the standard library. Create a directory for your environment, create the environment and activate it. Then install the package and all dependencies.

```bash
mkdir my_env_name
python -m venv my_env_name
my_env_name/Scripts/activate
pip install flexcv
```

Now you have installed everything you need to perform flexible cross validation and machine learning on your tabular data.

