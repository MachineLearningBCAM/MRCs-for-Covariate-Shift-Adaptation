# Double-Weighting for General Covariate Shift Adaptation
![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](#support-and-authors)

This repository is the official implementation of [Double-Weighting for General Covariate Shift Adaptation](https://arxiv.org/abs/2305.08637). 

The algorithm proposed in the paper provides efficient learning for the proposed Doble-Weighting for General Covariate Shift (DW-GCS). In particular, we first compute weights $\alpha$ and $\beta$ by solving the Double-Weighting Kernel Mean Matching (DW-KMM). Then, we learn the classifier's parameters by solving the MRC using double-weighting.

## Requirements

The standard libraries required are listed in the file `requirements.txt`. To install these libraries using

1) `pip`
```setup
pip install -r requirements.txt
```

2) `conda` environment
```
conda create --name <environment_name> --file requirements.txt
```

In addition, the implementation of the proposed algorithm utilizes the MOSEK optimizer for which license can be downloaded from [here](https://www.mosek.com/products/academic-licenses/).
