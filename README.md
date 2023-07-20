# Double-Weighting for General Covariate Shift Adaptation
![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg) [![Made with!](https://img.shields.io/badge/Made%20with-MATLAB-red)](/Matlab_Code)[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](#support-and-authors)

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

## Data

The repository contains multiple datasets to perform experiments as follows - 

Dataset | Covariates | Samples | Ratio of majority class | $\sigma$
--- | --- | --- | --- | ---
Blood | 3 | 748 | 76.20% | 0.7491
BreastCancer | 9 | 683 | 65.01% | 1.6064
Haberman | 3 | 306 | 75.53% | 1.3024
Ringnorm | 20 | 7400 | 50.49% | 3.8299
comp vs sci | 1000 | 5309 (tr) / 3534 (te) | 55.31% | 23.5628
comp vs talk | 1000 | 4888 (tr) / 3256 (te) | 60.06% | 23.4890
rec vs sci | 1000 | 4762 (tr) / 3169 (te) | 50.17% | 24.5642
rec vs talk | 1000 | 4341 (tr) / 2891 (te) | 55.02% | 25.1129
sci vs talk | 1000 | 4325 (tr) / 2880 (te) | 54.85% | 24.8320

The datasets are available as csv files in this repository in the `Datasets` and `Datasets_20News`folders.

## Support and Authors

José I. Segovia-Martín

jsegovia@bcamath.org

Santiago Mazuelas 

smazuelas@bcamath.org

Anqi Liu

aliu@cs.jhu.edu

## Citation

If you find useful the code in your research, please include explicit mention of our work in your publication with the following corresponding entry in your bibliography:

[1] José I. Segovia-Martín, S. Mazuelas, A. Liu "Double-Weighting for Covariate Shift Adaptation". Proceedings of the 40th International Conference on Machine Learning. PMLR, 2023.

The corresponding BiBTeX citation is given below:

@inproceedings{SegMazLiu2023,
  title={Double-Weighting for Covariate Shift Adaptation},
  author={Segovia-Mart'{i}n, Jos'{e} I. and Mazuelas, Santiago and Liu, Anqi},
  booktitle={International Conference on Machine Learning},
  pages={30439--30457},
  year={2023},
  organization={PMLR}
}
