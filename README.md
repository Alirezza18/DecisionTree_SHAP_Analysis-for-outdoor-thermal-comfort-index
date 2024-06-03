Decision Tree Classifier with Bayesian Optimization and SHAP Analysis
This repository contains code for training a Decision Tree Classifier using Bayesian Optimization for hyperparameter tuning and analyzing feature importance using SHAP (SHapley Additive exPlanations).

Overview
The code performs the following tasks:

Data Loading and Preprocessing: Loads UTCI data from a CSV file and preprocesses it by selecting relevant features and splitting it into training and testing sets.

Decision Tree Classifier: Defines a Decision Tree Classifier function (dt_classifier) that takes hyperparameters (max_depth, min_samples_split, min_samples_leaf) as inputs and evaluates the classifier's performance using cross-validation.

Bayesian Optimization: Utilizes Bayesian Optimization to find the optimal hyperparameters for the Decision Tree Classifier.

Model Training: Trains a Decision Tree Classifier using the best hyperparameters obtained from Bayesian Optimization.

SHAP Analysis: Computes SHAP values to interpret the trained model's predictions and visualizes the feature importance using summary plots.
