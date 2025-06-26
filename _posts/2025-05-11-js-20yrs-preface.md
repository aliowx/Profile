---
layout: post
title: "End-to-End Machine Learning about health care"
date: 2025-05-11
categories: [Machine learning, Deep Learning]

---
# 🧠 End-to-End Machine Learning Project Report

## 📌 Project Overview

This project involved building and evaluating machine learning models to predict


The full ML lifecycle was covered, including data preprocessing, modeling, evaluation, and explainability.

---

## 🛠️ Workflow Summary

### ✅ Data Preprocessing
- Handled missing values using `IterativeImputer`.
- Removed outliers with a custom `OutlierClipper`.
- Scaled numerical features using `StandardScaler`.
- Selected top features via `SelectKBest` with `f_regression`.

### ✅ Data Splitting
- Split into **train (68%)**, **validation (17%)**, and **holdout (15%)** sets using `train_test_split`.
- Ensured **no data leakage** by fitting preprocessors only on training data within the pipeline.

---

## 🤖 Model Training & Evaluation

The following models were trained using **Repeated K-Fold Cross-Validation**:
- Linear Models: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- Tree-Based: `RandomForest`, `ExtraTrees`, `GradientBoosting`, `XGBoost`
- Others: `KNN`, `MLPRegressor`

### 📊 Metrics Evaluated
- **R² Score**
- **MAE**
- **RMSE**

All results were compiled and visualized to compare model performances.