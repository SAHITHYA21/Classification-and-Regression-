# Linear and Quadratic Discriminant Analysis with Ridge Regression

This project implements Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and Ridge Regression from scratch in Python to classify and predict data outcomes while optimizing models for accuracy.

## Overview

The project focuses on:
- Developing LDA and QDA models for classification tasks.
- Implementing Ordinary Least Squares (OLS) and Ridge Regression for regression tasks.
- Exploring the effects of regularization and polynomial feature mapping on model performance.

## Key Features

### 1. Linear and Quadratic Discriminant Analysis
- **LDA**: Calculates class means and a shared covariance matrix to classify test data.
- **QDA**: Uses class-specific covariance matrices for classification.
- **Accuracy Evaluation**: Includes prediction accuracy and decision boundary visualization.

### 2. Ridge Regression
- **OLS Regression**: Minimizes Mean Squared Error (MSE) without regularization.
- **Ridge Regression**: Adds a regularization term controlled by a hyperparameter (`lambda`) to reduce overfitting.
- **Polynomial Mapping**: Maps features to higher-degree polynomials for non-linear regression tasks.

### 3. Optimization
- Optimizes Ridge Regression using direct minimization and gradient descent (`scipy.optimize.minimize`).
- Compares performance across regularization and polynomial degrees.

## Implementation Details

### Functions
- **`ldaLearn`** and **`qdaLearn`**: Train LDA/QDA models by computing class means and covariance matrices.
- **`ldaTest`** and **`qdaTest`**: Test LDA/QDA models on unseen data and compute accuracy.
- **`learnOLERegression`**: Implements Ordinary Least Squares regression.
- **`learnRidgeRegression`**: Implements Ridge Regression with a regularization term.
- **`mapNonLinear`**: Maps features to polynomial degrees for non-linear regression tasks.

### Visualization
- Decision boundaries for LDA and QDA.
- MSE curves for Ridge Regression across varying regularization strengths.
- Polynomial degree optimization for both regularized and unregularized models.

## Getting Started

### Prerequisites
Install the required libraries:
```bash
pip install numpy scipy matplotlib
