# LLM-Derived Features for Uncertainty-Aware Regression

## Project overview

This repository contains the Python code developed for my Master of Applied Technologies research in Cybersecurity and Networking.

The research examined whether structured features extracted from unstructured text using Large Language Models (LLMs) could improve the reliability of regression predictions. The extracted features were incorporated into probabilistic deep learning models, to evaluate their effect on prediction reliability and uncertainty estimation to be evaluated.

## Research objective

The study investigated whether LLM-derived thematic features could provide a richer representation of textual data than the original structured variables alone.

Two experimental settings were compared:

* Baseline models trained using the original structured numerical inputs
* Enhanced models trained using both structured numerical inputs and LLM-derived features

The results from these settings were compared to assess the effect of LLM-based feature extraction on prediction reliability and uncertainty quantification.

## Methodology

The research framework involved the following stages:

1. Preparing and preprocessing structured and unstructured data
2. Extracting thematic features from review text using LLMs
3. Converting the extracted information into structured numerical features
4. Training baseline and LLM-enhanced deep learning models
5. Estimating predictive uncertainty
6. Comparing model performance across the experimental settings
7. Interpreting model predictions using SHAP and LIME

The LLM-derived features represented:

* Feature presence
* Feature polarity
* Feature intensity

## Models

The study evaluated three uncertainty-aware deep learning approaches:

* Bayesian Neural Networks
* Monte Carlo Dropout
* Deep Ensembles

These models were used to generate regression predictions and quantify predictive uncertainty.

## Datasets

The experiments used five text-based datasets from different application areas:

* Flipkart product reviews
* Women’s clothing reviews
* Skincare product reviews
* Stress-related text data

The target variables included product ratings and confidence scores, depending on the dataset.

## Explainability

SHAP and LIME were used to examine how individual features contributed to the model predictions.

* **SHAP** was used to analyse feature importance across multiple observations.
* **LIME** was used to interpret selected individual predictions.

These methods supported the analysis of both the original variables and the features extracted using LLMs.

## Repository contents

The repository includes Python scripts for:

* LLM-based feature extraction
* Baseline experiments
* LLM-enhanced model training
* Deep Ensemble modelling
* SHAP analysis
* LIME analysis

The current filenames reflect the datasets, model types and experimental stages used during the research implementation.

## Technologies

The implementation was developed using:

* Python
* Pytorch
* Pandas
* NumPy
* Scikit-learn
* SHAP
* LIME
* Matplotlib

## Reproducibility

A fixed random seed of `42` was used where applicable. The datasets were divided into training and testing sets using an 80:20 split.

Local file paths within the scripts may need to be changed before running the experiments on another computer. Users must also obtain the relevant datasets separately if they are not included in this repository.

## Important note

The code in this repository was developed for academic research. Some scripts represent different stages of experimentation and may require changes to file paths, dataset names or software dependencies before execution.

## Author

**Raaliya Imran**
Master of Applied Technologies
Specialisation in Cybersecurity and Networking

## Acknowledgements

This research was completed under the supervision of **Dr Iman Ardekani** .

## Citation

If you use or refer to this work, please acknowledge the repository and the related Master’s research.

## Licence

No licence has currently been assigned to this repository. Unless a licence is added, the code remains protected by copyright and cannot automatically be reused, modified or redistributed by others.
