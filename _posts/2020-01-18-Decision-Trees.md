---
title: "Decision Tree Basic"
date: 2020-01-18
categories: Machine-Learning
---

[TOC]

# Modeling

## Visulizing


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
print("features:", iris['feature_names'])
print("targets:", iris['target_names'])
X = iris.data[:, 2:] # petal length and width
y = iris.target
```

    features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    targets: ['setosa' 'versicolor' 'virginica']

```python
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)
```


```python
import os
from sklearn import tree
from subprocess import call
from IPython.display import Image
from sklearn.tree import export_graphviz

IMAGES_PATH = "imgs"
tree.export_graphviz(
    tree_clf,
    out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True)
call(['dot', '-T', 'png', 'imgs/iris_tree.dot', '-o', 'imgs/iris_tree.png'])
display(Image("imgs/iris_tree.png"))
```


![png](imgs/iris_tree.png)


In particular, they don’t require feature scaling or centering at all.

## Class Probabilities


```python
print("Probabilities:", tree_clf.predict_proba([[5, 1.5]]))
print("Prediction:", tree_clf.predict([[5, 1.5]]))
```


# CART (Classification)

> This is a greedy algorithm: it greedily searches for an optimum split at the top level, then repeats the process at each level.

Scikit-Learn uses the Classification And Regression Tree (CART) algorithm to train Decision Trees (also called “growing” trees). It searches for the pair $(k, t_k)$ (e.g. petal length < 2.45cm) that produces the purest subsets (weighted by their size).

## Cost function

$$
J(k,t_k)=\frac{m_{left}}{m}G_{left}+\frac{m_{right}}{m}G_{right}
$$

G is the impurity of the subset, m is the number of instances in the subset

## Computational Complexity

- Training: $O(n\times m log(m))$
    - set `presort=True` less than a few thousand instances can speed up, but slows down when large
- Prediction: $O(log_2(m))$

## Criterion

### Gini

A node’s gini attribute measures its impurity: a node is “pure” (gini=0) if all training instances it applies to belong to the same class.

$$
G_i=1-\sum^n_{k=1}{p_{i,k}^2}
$$

$p_{i,k}$is the ratio of class k instances among the training instances in the $i^{th}$ node.

e.g. $G_2=1-((0/54)^2+(49/54)^2+(5/54)^2)=0.168$

### Entropy

The concept of entropy originated in thermodynamics as a measure of molecular disorder: entropy approaches zero when molecules are still and well ordered. Entropy is zero when all messages are identical.

$$
H_i=-\sum^n_{k=1\\p_{i,k}\ne0}log_2(p_{i,k})
$$

e.g. $H_2=-((49/54)log_2(49/54)+(5/54)log_2(5/54))=0.445$

### Compare

- Gini: faster, tends to isolate the most frequent class in its own branch of the tree
- Entropy: tends to produce slightly more balanced trees

# CART (Regression)

```python
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2) 
tree_reg.fit(X, y)
```

## Cost function

$$
x J(k,t_k)=\frac{m_{left}}{m}MSE_{left}+\frac{m_{right}}{m}MSE_{right}\\\hat{y}_{node}=\frac{1}{m_{node}}\sum{y^{(i)}}
$$

# Regularization Hyperparameters

To avoid overfitting the training data, you need to restrict the Decision Tree’s freedom during training.

- `max_depth` hyperparameter (the default value is None, which means unlimited). Reducing max_depth will regularize the model and thus reduce the risk of overfitting.
- `min_samples_split` (the minimum number of samples a node must have before it can be split).
- `min_samples_leaf` (the minimum number of samples a leaf node must have)
- `min_weight_fraction_leaf` (same as min_samples_leaf but expressed as a fraction of the total number of weighted instances)
- `max_leaf_nodes` (maximum number of leaf nodes)
- `max_features` (maximum number of features that are evaluated for splitting at each node) 

> Increasing `min_*` hyperparameters or reducing `max_*` hyperparameters will regularize the model.
