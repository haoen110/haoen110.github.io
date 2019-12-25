---
title: "Machine Learning Concepts"
date: 2019-12-25
categories: Machine-Learning
---

# What is Machine Learning?
Machine Learning is the science of programming computers so they can learn from data to improve the performance on some tasks.

# Types of Machine Learning

## 1. From Human Supervision

### 1.1 Supervised Learning
In supervised learning, the training data you feed to the algorithm includes the desired solutions, called labels. 

- k-Nearest Neighbors 
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVMs)
- Decision Trees and Random Forests
- Neural networks
    - Unsupervised: autoencoders, restricted Boltzmann machines
    - Semisupervised, deep belief networks, unsupervised pretraining

### 1.2 Unsupervised Learning

In unsupervised learning, the training data is unlabeled. The system tries to learn without a teacher.   

- Clustering 
    - K-Means 
    - DBSCAN 
    - Hierarchical Cluster Analysis (HCA)
- Anomaly detection and novelty detection 
    - One-class SVM 
    - Isolation Forest
- Visualization and dimensionality reduction
    - Principal Component Analysis (PCA)
    - Kernel PCA
    - Locally-Linear Embedding (LLE)
    - t-distributed Stochastic Neighbor Embedding (t-SNE)
- Association rule learning
    - Apriori
    - Eclat
    
### 1.3 Semisupervised Learning
Those algorithms can deal with partially labeled training data. Such as photo tagging function in google photo and apple icloud photo.

### 1.4 Reinforcement Learning
The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards in return. Such as alpha go.

## 2. Whether Learn Incrementally on the Fly

### 2.1 Batch learning (Offline Learning)
The system is incapable of learning incrementally: it must be trained using all the available data offline then applies what it has learned.

### 2.2 Online Learning
Train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches.
- out of core learning: Train systems on huge datasets that cannot fit in one machine’s main memory (offline).
- Challenge: data quality.

## 3. How they Generalize

### 3.1 Instance-based Learning
The system learns the examples by heart, then generalizes to new cases by comparing them to the learned examples using a similarity measure. Such as kNN.

### 3.2 Model-based Learning
Build a model of these examples, then use that model to make predictions.

# Main Challenges of Machine Learning

- Insufficient Quantity of Training Data
- Nonrepresentative Training Data (Sampling Bias)
- Poor-Quality Data
- Irrelevant Features
- Overfitting the Training Data
  - Regularization: simplify the model (fewer parameters; reducing attributes)
  - More training data
  - Reducing the noise

- Underfitting the Training Data
  - Contrast with above

# Testing and Validating

## Hold out Validation

Hold out part of the training set to evaluate several candidate models with different hyperparameters and select the best one. 

1. Train multiple models with various hyperparameters on the reduced training set (i.e., the full training set minus the validation set)
2. Select the model that performs best on the validation set.
3. Train the best model on the full training set (including the validation set), and this gives the final model.

However, whether the validation set is too small or too large, the model evaluations will be imprecise.

## Cross-Validation

Each model is evaluated once per validation set, after it is trained on the rest of the data. By averaging out all the evaluations of a model, we get a much more accurate measure of its performance. However, there is a drawback: the **training time** is multiplied by the number of validation sets.

## Data Mismatch

In some cases, it is easy to get a large amount of data for training, but it is not perfectly representative of the data that will be used in production. For example, using pictures online train the model to apply in test set that images took by phone.