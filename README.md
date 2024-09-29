# Machine Learning Projects

This repository contains various machine learning projects implemented using Python. The projects explore different algorithms and techniques, including K-Nearest Neighbors, Logistic Regression, Multi-Layer Perceptron, and applications in Regression and Natural Language Processing.

## Projects Overview

### 1. K-Nearest Neighbour.ipynb
   - Implementation of the K-Nearest Neighbors algorithm for classification tasks.

### 2. Logistic Regression.ipynb
   - **Task/Problem Statement**: The goal of this assignment is to solve multi-class classification problems using the Logistic Regression model and to use visualization techniques for analyzing errors.
   
   - **Part A**: Multi-class Classification – Structured Data
   
   - **Part B**: Multi-class Classification – Unstructured Data & Analysis of Model

   - **Dataset**: The dataset, given in the `winequality-white.csv` file, relates to the white variants of the Portuguese “Vinho Verde” wine. It provides the physicochemical (inputs) and sensory (output) variables. The dataset consists of characteristics of white wine (e.g., alcohol content, density, amount of citric acid, pH, etc.), with the target variable “quality” representing the wine rating. The target variable “quality” ranges from 3 to 9, where a higher rating indicates better wine quality. The classes are ordered and not balanced (e.g., there are many more normal wines than excellent or poor ones).

   - **Input Variables**:
     11 Wine features
   
   - **Output Variable**: 
     - “Quality” representing the rating of wine, ranging from 3 to 9.


   - **Key Techniques Used**:
     - Libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`
     - Algorithms: Logistic Regression
     - Techniques: Cross-validation, Grid Search for hyperparameter tuning, confusion matrix analysis

### 3. Logistic_Regression.ipynb
   - Another implementation of Logistic Regression, providing additional insights and examples.

### 4. Multi Layer Perceptron.ipynb
   - **Task/Problem Statement**: The goal of this assignment is to solve multi-class classification problems using the Multi-Layer Perceptron (MLP) model and learn how to train this model optimally.
   
   - **Part A**: An extensive study of the MLP model
   
   - **Part B**: Design an MLP for optimal performance

   - **Dataset**: The MNIST (Modified National Institute of Standards and Technology) dataset consists of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau. Each image is labeled with the digit it represents.

   - **Description**: 
     - The dataset contains 70,000 images, each a grayscale 28 x 28 pixel image. Each feature represents one pixel’s intensity, ranging from 0 (white) to 255 (black). The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

   - **Source**: Load this dataset using the Keras API: [MNIST Dataset](https://keras.io/api/datasets/mnist/)

   - **Input Variables**: 
     - Small images of handwritten digits.
   
   - **Output Variable**: 
     - 10 classes representing integer values from 0 to 9.

   - **Key Techniques Used**:
     - Libraries: `numpy`, `pandas`, `matplotlib`, `keras`
     - Algorithms: Multi-Layer Perceptron
     - Techniques: Model design and training for optimal performance

### 5. Regression & Natural Language Processing using Linear Regression & K-Nearest Neighbors.ipynb
   - **Task/Problem Statement**: The goal of this assignment is to solve regression and classification problems using the following models on two types of data: numeric and text.
   
   - **Part A**: Numeric Data - Regression Problem – Linear Regression using the Stochastic Gradient Descent algorithm
   
   - **Part B**: Natural Language Processing - Text Data - Classification Problem – K-Nearest Neighbors

   - **Dataset**: The energy efficiency dataset, `EnergyEfficiency.xlsx`, is created to perform energy analysis.

   - **Description**: 
     - The dataset comprises 768 samples and 8 features (X1 to X8). It has two real-valued target variables (Y1 and Y2), i.e., heating load and cooling load, respectively.

   - **Input Variables**:
     - X1: Relative Compactness
     - X2: Surface Area
     - X3: Wall Area
     - X4: Roof Area
     - X5: Overall Height
     - X6: Orientation
     - X7: Glazing Area
     - X8: Glazing Area Distribution

   - **Output Variables**: 
     - Y1: Heating Load
     - Y2: Cooling Load

   - **Source**: Not specified (provide the source if available).

   - **Key Techniques Used**:
     - Libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`
     - Algorithms: Linear Regression, K-Nearest Neighbors
     - Techniques: Stochastic Gradient Descent for linear regression, classification using KNN

## Installation

To run the notebooks, ensure you have Python and Jupyter Notebook installed. You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib keras
