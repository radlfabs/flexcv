# Getting Started with flexcv

This guide will help you get started with the `flexcv` package quickly. It will show you how to install the package and how to use it to compare machine learning models on your data.

## Installation

Clone our [repository on github](https://github.com/radlfabs/flexcv) to your computer.

`flexcv` was tested using Python v3.10. You can easily install this version of Python using conda (Anaconda or Miniconda). We recommend using a fresh environment for cleanly holding all relevant packages corresponding to this repo. With conda installed, you can create a new Python 3.10 environment, activate it, and install our requirements by running the following lines from the command line:

```bash
conda create --n flexcv python=3.10
conda activate flexcv
conda install pip
cd path/to/this/repo
pip install -r requirements.txt
```

Now you have installed everything you need to perform flexible cross validation and machine learning on your tabular data.

## Getting Started

To get started with `flexcv`, you can import it into your Python script or notebook:

```py
from flexcv import CrossValidation
```

You can then use the various configuration classes provided by the package to compare machine learning models on your data.
