# Weddell Gyre Clusters

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
 
 In this project, we use unsupervised classification to examine structures in the Weddell Gyre region dataset prepared by Shenjie Zhou (BAS) as part of the SO-CHIC project. 

## Getting started

### Docker container
At present, you can use Docker to get this project installed and running. These commands have been tested and should work:
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

### Singularity container
If you're working on an HPC platform (e.g for access to more memory or GPUs), you'll probably need to run singularity instead. These commands have been tested on a specific platform at BAS:
```
$ cd <where-you-want-to-work>
$ singularity pull docker://pangeo/pangeo-notebook:2021.09.08
$ singularity shell --bind <<<YOUR DATA DIRECTORY>>:/mnt pangeo-notebook_2021.09.08.sif -nv 
Singularity> source activate notebook
(notebook) Singularity> pip install seaborn gsw umap-learn cmocean
(notebook) Singularity> ipython
cd /mnt
```
The `-nv` flag should give access to GPUs. After the `source activate notebook` step, you might have to pip install gsw, seaborn, and umap-learn

### Downloading the larger South Atlantic dataset

Although this repository focuses on a set of profiles near the Antarctic, the initial dataset covers the South Atlantic and some of the Indian Ocean as well. This larger dataset (~500MB) has been archived and is accessible via Zenodo (doi:10.5281/zenodo.7465132). The code expects to find this file in `weddell_gyre_clusters/src/models/`. It is not strictly necessary to download this dataset to reproduce most of the figures in this repository; most of the figures can be reproduced without any additional downloads beyond this GitHub repository itself.

### Expected directory structure

The expected directory structure:
```
├── so-chic-data            <- Contains profile data as processed using MITprof, before it is classified
└── weddell_gyre_clusters   <- This current repository
```

## Project Organization
```
├── Dockerfile         <- Docker file that can be used for containerised computing
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── requirements.txt   <- Package requirements (you might be able to make a conda env with this; hasn't been tested)
└── src                <- Source code, including jupyter notebooks and other functions
```

## Acknowledgement

This work was funded by a UKRI Future Leaders Fellowship (MR/T020822/1)

---

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).
