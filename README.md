# Weddell Gyre Clusters

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
 
 In this project, we use unsupervised classification to examine structures in the Weddell Gyre region dataset prepared as part of the SO-CHIC project

## Requirements
- Python 3.8+

## Getting started

### Docker container
At present, you can use Docker to get this project installed and running. (We could go with a simpler docker container in the future). These commands work for me:
```
cd <where-you-want-to-work>
docker pull pangeo/pangeo-notebook:2021.09.08
docker run --rm -it -p 8888:8888 -v $PWD:/work -w /work pangeo/pangeo-notebook:2021.09.08 jupyter-lab --no-browser --ip=0.0.0.0
```
Note that Docker can only see "down the tree", so make sure that any data and files that you want to work with are within the tree. To test the source code in `src`, you can use this same Docker image to run `ipython`:
```
docker run --rm -it -p 8808:8808 -v $PWD:/work -w /work pangeo/pangeo-notebook:2021.09.08 bash
```
and then run `ipython3`. 

### Singularity container
Alternatively, if you're working on an HPC platform, you'll probably need to run singularity instead. These commands for reference:
```
cd <where-you-want-to-work>
singularity pull docker://pangeo/pangeo-notebook:2021.09.08
singularity run --bind /data/expose/so-wise/:/mnt pangeo-notebook_2021.09.08.sif bash
cd /mnt
```
The `--nv` flag turns on GPU support, which may not be available on every workstation. 

## Project Organization
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make init` or `make lint-requirements`
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
|   |                     the creator's initials, and a short `-` delimited description, e.g.
|   |                     `1.0_jqp_initial-data-exploration`.
│   ├── exploratory    <- Notebooks for initial exploration.
│   └── reports        <- Polished notebooks for presentations or intermediate results.
│
├── report             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── sections       <- LaTeX sections. The report folder can be linked to your overleaf
|                         report with github submodules.
│
├── requirements       <- Directory containing the requirement files.
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data_loading   <- Scripts to download or generate data
│   │
│   ├── preprocessing  <- Scripts to turn raw data into clean data and features for modeling
|   |
│   ├── models         <- Scripts to train models and then use trained models to make
│   │                     predictions
│   │
│   └── tests          <- Scripts for unit tests of your functions
│
└── setup.cfg          <- setup configuration file for linting rules
```

## Code formatting
To automatically format your code, make sure you have `black` installed (`pip install black`) and call
```black . ``` 
from within the project directory.

---

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).
