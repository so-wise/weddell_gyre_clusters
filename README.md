# Weddell Gyre Clusters

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
 
 In this project, we use unsupervised classification to examine structures in the Weddell Gyre region dataset prepared as part of the SO-CHIC project

## Requirements
- Python 3.8+
- gsw==3.4.0
- xarray==0.19.0
- xgcm==0.5.2
- Cartopy==0.19.0.post1
- joblib==1.0.1
- matplotlib==3.4.3
- numpy==1.21.2
- pandas==1.3.2
- dask==2021.11.1
- scikit_learn==0.24.2
- seaborn==0.11.2
- umap==0.1.1

## Getting started

### Docker container
At present, you can use Docker to get this project installed and running. (We could go with a simpler docker container in the future). These commands work for me:
```
cd <where-you-want-to-work>
docker pull dannes/weddell-clusters
docker run --rm -it -p 8888:8888 -v $PWD:/work -w /work dannes/weddell-clusters jupyter-lab --no-browser --ip=0.0.0.0
```
Note that Docker can only see "down the tree", so make sure that any data and files that you want to work with are within the tree. To test the source code in `src`, you can use this same Docker image to run `ipython`:
```
docker run --rm -it -v $PWD:/work -w /work dannes/weddell-clusters bash
ipython3
```

### Singularity container (NOT WORKING AT PRESENT)
NOTE: I'm having a shell issue. I have to run `source .cshrc` when I log on in order for my aliases to be defined. 

If you're working on an HPC platform (e.g for access to more memory or GPUs), you'll probably need to run singularity instead. These commands seem to basically work on BAS HPC:
```
$ cd <where-you-want-to-work>
$ singularity pull docker://pangeo/pangeo-notebook:2021.09.08
$ singularity shell --bind /data/expose/so-wise/:/mnt pangeo-notebook_2021.09.08.sif -nv 
Singularity> source activate notebook
(notebook) Singularity> pip install seaborn gsw umap-learn cmocean
(notebook) Singularity> ipython
cd /mnt
```
The `-nv` flag should give access to GPUs. After the `source activate notebook` step, you might have to pip install gsw, seaborn, and umap-learn. Eventually I need to get these into their own image, once I learn how to do that. 

(Now, I'm running into lots of other bugs, but they don't seem to be related to the computing environment.)

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
