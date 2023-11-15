This guide will help you get started with the `flexcv` package quickly. It will show you how to install the package and how to use it to compare machine learning models on your data. At the end of this section you will have a working environment to start using `flexcv` for your own projects.

## Installation

Clone our [repository on github](https://github.com/radlfabs/flexcv) to your computer.

##### Usind conda

`flexcv` was tested with Python 3.10 and 3.11. You can easily install this version of Python using conda (Anaconda or Miniconda). We recommend using a fresh environment for cleanly holding all relevant packages corresponding to this repo. With conda installed, you can create a new Python 3.11 environment, activate it, and install our requirements by running the following lines from the command line:

```bash
conda create --n flexcv python=3.11
conda activate flexcv
conda install pip
cd path/to/this/repo
pip install -r requirements.txt
```

##### Using venv

To separate Python environments on your system, you can also use the `venv` package from the standard library.

```bash
cd path/to/this/repo
python -m venv my_env_name
my_env_name/Scripts/activate
pip install flexcv/requirements.txt
```

Now you have installed everything you need to perform flexible cross validation and machine learning on your tabular data.

