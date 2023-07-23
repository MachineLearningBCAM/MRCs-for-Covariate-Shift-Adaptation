# Double-Weighting for General Covariate Shift Adaptation
[![Made with!](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](#python-code) [![Made with!](https://img.shields.io/badge/Made%20with-MATLAB-red)](#matlab-code) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](#support-and-authors)

This repository is the official implementation of [Double-Weighting for General Covariate Shift Adaptation](https://arxiv.org/abs/2305.08637). 

The algorithm proposed in the paper provides efficient learning for the proposed Doble-Weighting for General Covariate Shift (DW-GCS). In particular, we first compute weights $\alpha$ and $\beta$ by solving the Double-Weighting Kernel Mean Matching (DW-KMM). Then, we learn the classifier's parameters by solving the MRC using double-weighting.

## Python Code

### Requirements

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

### Code

The functions in the folder [Python_code](https://github.com/MachineLearningBCAM/MRCs-for-Covariate-Shift-Adaptation/tree/main/Experiments_Paper) folder contains experiment scripts required to replicate the experiments of the paper.

* Experiment_Features.py, Experiment_PCA.py and Experiment_20News.py are the main files. 
* Experiment_Features.py file perform a experiment for datasets in which we artificially introduce the covariate shift based on the median of the features. 
* Experiment_PCA.py file perform a experiment for datasets in which we artificially introduce the covariate shift based on the median of the first principal component of the features. 
* Experiment_20News.py file perform a experiment using the “News20groups” dataset that is intrinsically affected by a covariate shift since the training and testing partitions correspond to different times.
* In such files we can modify the feature mapping, and the loss we are going to use. We can also choose if the classifier is deterministic or not.
* The file CovShiftGen.py generate covariate shift in the datasets. In particular, we select training and testing samples with different probabilities based on the medians of the first 3 features, and based on the median of the first principal component of features.
* The folder [Auxiliary_Functions](https://github.com/MachineLearningBCAM/MRCs-for-Covariate-Shift-Adaptation/tree/main/Auxiliary_Functions) contains the function phi.py that calculates the feature mappings using linear, polinomial or random Fourier features (RFF).
* The folder [DWGCS](https://github.com/MachineLearningBCAM/MRCs-for-Covariate-Shift-Adaptation/tree/main/DWGCS) contains the functions of the Double-Weighting for General Covariate Shift Adaptation.
* DWGCS.DWKMM computes the estimated weights $\beta$ and $\alpha$ solving the double-weighting kernel mean matching.
* DWGCS.parameters obtains mean vector estimate $\tau$ and confidence vector $\lambda$.
* DWGCS.learning solves the convex MRC optimization problem using double-weighting and obtains the classifier parameters.
* DWGCS.prediction assigns labels to instances and gives the classification error.

## Matlab Code

[Matlab_code](https://github.com/MachineLearningBCAM/MRCs-for-Covariate-Shift-Adaptation/tree/main/Matlab_Code%20) folder contains Matlab scripts required to execute the method:

* run_DWGCS_example1.m and run_DWGCS_example1.m are the main files. 
* run_DWGCS_example1.m file perform a experiment for datasets in which we artificially introduce the covariate shift. 
* run_DWGCS_example2.m file perform a experiment using the “News20groups” dataset that is intrinsically affected by a covariate shift since the training and testing partitions correspond to different times. 
* In such files we can modify the feature mapping, and the loss we are going to use. We can also choose if the classifier is deterministic or not.
* The functions in the folder [CovShift_Generation](https://github.com/MachineLearningBCAM/MRCs-for-Covariate-Shift-Adaptation/tree/main/Matlab_Code%20/CovShift_Generation) generate covariate shift in the datasets. In particular, we select training and testing samples with different probabilities based on the medians of the first 3 features, and based on the median of the first principal component of features.
* phi.m calculates the feature mappings using linear, polinomial or random Fourier features (RFF).
* DWGCS_weights.m computes the estimated weights $\beta$ and $\alpha$ solving the double-weighting kernel mean matching.
* DWGCS_parameters.m obtains mean vector estimate $\tau$ and confidence vector $\lambda$.
* DWGCS_learning.m solves the convex MRC optimization problem using double-weighting and obtains the classifier parameters.
* DWGCS_prediction.m assigns labels to instances and gives the classification error.

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