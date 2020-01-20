---
title: "Inro Models"
date: 2020-01-18
categories: Machine-Learning
---

[TOC]

# Linear Regression

## Model

> n is the number of features
>
> m is the number of inputs

- Model: $\hat{y}=\theta_0 + \theta_1 x_1 +\theta_2 x_2 +...+\theta_n x_n$

- Model (vectorized form): $\hat{y}=h_{\pmb{\theta}}(\pmb{x})=\pmb{\theta \cdot x}$, if $\theta$ is a column vector then, $\hat{y}=h_{\pmb{\theta}}(\pmb{x})=\pmb{\theta^T \cdot x}$

## Cost function

$$
MSE=\frac{1}{m}\sum^m_{i=1}{(\theta^Tx^i-y^i)^2}
$$

## Estimation

### The Normal Equation

- Equation: $\hat{\theta}=(X^TX)^{-1}X^Ty$
- Complexity: $O(n^{2.4})$ to $O(n^{3})$
- Evaluation: Fast for large m, slow for large n

### By Singular Value Decomposition Approach

- Equation: $ \hat{\theta}=X^+y$, where $X^+=V\Sigma^+U^T$
- Complexity: $O(n^{2})$
- Evaluatioun: Fast for large m, slow for large n

This approach is more **efficient** than computing the Normal Equation, plus it handles edge cases nicely: indeed, the Normal Equation may not work if the matrix $X^TX$ is not invertible (i.e., **singular**), such as if **m < n** or if some features are redundant, but the **pseudoinverse** is always defined.

# Gradient Descent

Gradient Descent is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Descent is to <u>tweak parameters</u> **iteratively** in order to minimize a cost function.

### Base hyperarameters

- Learning rate: Size of steps

  - Too small: The algorithm will have to go through many iterations to converge, which will take a long time
  - Too high: This might make the algorithm **diverge**, with larger and larger values, failing to find a good solution

- Random initialization: Start by filling $\theta$ with random values

  - Bad choice of initialization may lead to the **local minimum**, which is not as good as the **global minimum**.

  - > Fortunately, the **MSE cost function** for a **Linear Regression** model happens to be a **convex function**, which means that if you pick any two points on the curve, the line segment joining them never crosses the curve. This implies that there are no local minima, just one global minimum.

### Scale

When using Gradient Descent, we should ensure that all features have a **similar scale** (e.g., using Scikit-Learn’s **StandardScaler** class), or else it will take much longer to converge.

## Batch Gradient Descent

> Features: use whole training set, fixed learning rate

### Equation

- Partial derivatives of the cost function: Calculate how much the cost function will change if you change $\theta_j$ just a little bit.

$$
\frac{\partial}{\partial\theta_j}MSE(\theta)=\frac{2}{m}\sum^m_{i=1}{(\theta^Tx^i-y^i)}x_j^i
$$

- Gradient vector of cost function
  $$
  \triangledown_\theta MSE(\pmb{\theta})=\frac{2}{m}X^T(X\theta-y)
  $$

- 

$$
\theta^{next step}=\theta-\eta\triangledown_\theta MSE(\pmb{\theta})
$$

### Algorithm

```python
eta = 0.1 # learning rate 
n_iterations = 1000 
m = 100
theta = np.random.randn(2,1) # random initialization
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) 
    theta = theta - eta * gradients
```

### Tolerance

Set a very large number of iterations but to interrupt the algorithm when the gradient vector becomes tiny—that is, when its norm becomes smaller than a tiny number $\epsilon$.

### Convergence Rate

Batch Gradient Descent with a **fixed** learning rate will eventually converge to the optimal solution, but you may have to wait a while: it can take $O(1/\epsilon)$ iterations to reach the optimum within a range of $\epsilon$ depending on the shape of the cost function.

## Stochastic Gradient Descent

> Features: use random instance, gradually reduce the learning rate

Stochastic Gradient Descent just picks a **random instance** in the training set at every step and computes the gradients based only on that single instance, which has a better chance of finding the **global minimum** than Batch Gradient Descent does (jump out of **local minima**).

### Reduce the Learning Rate

The function that determines the learning rate at each iteration is called the **learning schedule**. If the learning rate is reduced **too quickly**, you may get stuck in a **local minimum**, or even end up frozen halfway to the minimum. If the learning rate is reduced **too slowly**, you may jump around the minimum for a long time and end up with a suboptimal solution if you halt training too early. Iterate by rounds of m iterations; each round is called an **epoch**.

```python
n_epochs = 50 
t0, t1 = 5, 50 # learning schedule hyperparameters
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2,1) # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m) 
        xi = X_b[random_index:random_index+1] 
        yi = y[random_index:random_index+1] 
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi) 
        eta = learning_schedule(epoch * m + i) 
        theta = theta - eta * gradients
```

```python
# in sklearn
from sklearn.linear_model import SGDRegressor 
# max_iter is the number of epochs
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1) sgd_reg.fit(X, y.ravel())
```

## Mini-batch Gradient Descent

Minibatch GD computes the gradients on **small random sets of instances** called **minibatches**. The main advantage of Mini-batch GD over Stochastic GD is that you can get a performance boost from hardware optimization of matrix operations, especially when using **GPUs**.

## Comparison

| Algorithm       | Large m | Large n | Scaling | Scikit-Learn     |
| --------------- | ------- | ------- | ------- | ---------------- |
| Normal Equation | Fast    | Slow    | No      | n/a              |
| SVD             | Fast    | Slow    | No      | LinearRegression |
| Batch GD        | Slow    | Fast    | Yes     | SGDRegressor     |
| Stochastic GD   | Fast    | Fast    | Yes     | SGDRegressor     |
| Mini-batch GD   | Fast    | Fast    | Yes     | SGDRegressor     |

# Learning Curves

> If a model performs <u>**well** on the training data</u> but generalizes <u>**poorly** according to the cross-validation metrics</u>, then your model is **overfitting**; If it performs <u>poorly on both</u>, then it is **underfitting**. This can be viewed on learning curves.

```python
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], [] 
    for m in range(1, len(X_train)):
				model.fit(X_train[:m], y_train[:m])
				y_train_predict = model.predict(X_train[:m])
				y_val_predict = model.predict(X_val)
				train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
				val_errors.append(mean_squared_error(y_val, y_val_predict)) plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train") plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
lin_reg = LinearRegression() 
plot_learning_curves(lin_reg, X, y)
```

### Underfitting

If your model is underfitting the training data, adding more training examples will not help. You need to:

- use a more **complex model** 
- come up with **better features**.

### Overfitting

One way to improve an overfitting model is to:

- feed it more **training data** until the **validation error** reaches the **training error**.

## Generalization Errors

> **Increasing** a model’s complexity will typically <u>increase its **variance** and reduce its **bias**</u>. Conversely, **reducing** a model’s complexity <u>increases its **bias** and reduces its **variance**</u>.

### Bias

Due to **wrong assumptions**, such as assuming that the data is <u>linear</u> when it is actually <u>quadratic</u>. A <u>high-bias</u> model is most likely to **underfit** the training data.

### Variance

Due to the model’s **excessive sensitivity** to small variations in the training data. A model with many <u>degrees of freedom</u> (such as a high-degree polynomial model) is likely to have <u>high variance</u>, and thus to **overfit** the training data.

### Irreducible Error

This part is due to the **noisiness** of the data itself. The only way to reduce this part of the error is to **clean up** the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers).

## Early Stopping

After a while the validation error stops decreasing and actually starts to go back up. This indicates that the model has started to **overfit** the training data. With early stopping you just stop training as soon as the validation error reaches the minimum.

![image-20200114155514223](/Users/haoen110/Documents/My Projects/ml-concepts/imgs/early-stopping.jpg)

### Code

```python
from sklearn.base import clone
# prepare the data 
poly_scaler = Pipeline([
  	("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
		("std_scaler", StandardScaler()) ]) 
X_train_poly_scaled = poly_scaler.fit_transform(X_train) 
X_val_poly_scaled = poly_scaler.transform(X_val) 
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005) # warm_start=True, when the fit() method is called, it just continues training where it left off instead of restarting from scratch.
minimum_val_error = float("inf") 
best_epoch = None 
best_model = None

for epoch in range(1000):
		sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off 
    y_val_predict = sgd_reg.predict(X_val_poly_scaled) 
    val_error = mean_squared_error(y_val, y_val_predict) 
    if val_error < minimum_val_error:
				minimum_val_error = val_error
				best_epoch = epoch
        best_model = clone(sgd_reg)
```

Or

```python
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, early_stopping=True, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
```

# Regularized Linear Models

> Scale the data!

## Ridge Regression

> This algorithm not only fit the data but also keep the **model weights** as **small** as possible.

### Cost Function

$$
J(\theta)=MSE(\theta)+\alpha\sum^n_{i=1}\theta^2_i=MSE(\theta)+\alpha||w||_2
$$

where $||w||_2$ represents the $\ell_2$ of the weight vector. The hyperparameter $\alpha$ controls how much you want to regularize the model. If $\alpha=0$ then it is  just a **Linear Regression**; If $\alpha$ is very large, then all **weights** end up very close to **zero** and the result is a **flat line** going through the data’s mean.

- Increasing of $\alpha$ will **reduces** the model’s **variance** but **increases** its **bias**.

> Note that the **regularization term** should only be added to the cost function during **training**. Once the model is trained, you want to evaluate the model’s performance using the **unregularized performance measure**.

### Equation

$$
\hat{\theta}=(X^TX+\alpha A)^{-1}X^Ty
$$

where A is the (n + 1) × (n + 1) identity matrix except with a 0 in the top-left cell, corresponding to the bias term.

### Code

- sklearn

  ```python
  from sklearn.linear_model import Ridge
  # ‘cholesky’ uses the standard scipy.linalg.solve function to obtain a closed-form solution
  # ‘auto’ chooses the solver automatically based on the type of data
  ridge_reg = Ridge(alpha=1, solver="cholesky")
  ridge_reg.fit(X, y)
  ridge_reg.predict([[1.5]])
  ```

- SGD

  ```python
  from sklearn.linear_model import SGDRegressor
  sgd_reg = SGDRegressor(penalty="l2")
  sgd_reg.fit(X, y.ravel())
  sgd_reg.predict([[1.5]])
  ```

## Lasso Regression

> Least Absolute Shrinkage and Selection Operator Regression. This algorithm tends to completely eliminate the weights of the least important features (i.e., set them to zero).

### Cost Function

$$
J(\theta)=MSE(\theta)+\alpha\sum^n_{i=1}|\theta_i|=MSE(\theta)+\alpha||w||_1
$$

### Code

```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
```

## Elastic Net

> Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The regularization term is a simple mix of both Ridge and Lasso’s regularization terms, and you can control the mix ratio r. When r = 0, Elastic Net is equivalent to **Ridge Regression**, and when r = 1, it is equivalent to **Lasso Regression.**

### Cost Function

$$
J(\theta)=MSE(\theta)+r\alpha\sum^n_{i=1}|\theta_i|+\frac{1-r}{2}\alpha\sum^n_{i=1}\theta_i^2
$$

> **Elastic Net** is **preferred** over Lasso since Lasso may behave erratically when the  number of features is greater than the number of training instances or **when several features are strongly correlated**.

### Code

```python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
```

# Logistic Regression

If the estimated probability is greater than 50%, then the model predicts that the instance belongs to that class (called the positive class, labeled “1”), or else it predicts that it does not (i.e., it belongs to the negative class, labeled “0”). This makes it a binary classifier.

### Model

- Logistic Regression model estimated probability (vectorized form)

$$
\hat{p}=h_\theta(x)=\sigma(x^T\theta)
$$

- Logistic function

$$
\sigma(t)=\frac{1}{1+exp(-t)}
$$

![image-20200115110134245](/Users/haoen110/Documents/My Projects/ml-concepts/imgs/logit.png)

> The score **t** is often called the **logit**: this name comes from the fact that the logit function, defined as <u>logit(p) = log(p / (1 - p))</u>, is the inverse of the logistic function. 
>
> Indeed, if you compute the logit of the estimated probability p, you will find that the result is t. The logit is also called the **log-odds**, since it is the log of the ratio between the estimated probability for the positive class and the estimated probability for the negative class.

- Logistic Regression model prediction

$$
\hat{y}=\left\{
                \begin{array}{ll}
                  0\ \ if\ \hat{p}\lt0.5\\
                  1\ \ if\ \hat{p}\ge0.5\\
                \end{array}
              \right.
$$

### Cost Function

- Cost function of a single training instance

$$
c(\theta)=\left\{
                \begin{array}{ll}
                  -log(\hat{p})\ \ if\ \ \hat{y}=1\\
                  -log(1-\hat{p})\ \ if\ \ \hat{y}=0\\
                \end{array}
              \right.
$$

> – log(t) grows very large when t approaches 0, so the cost will be large if the model estimates a probability close to 0 for a positive instance, and it will also be very large if the model estimates a probability close to 1 for a negative instance.

- Logistic Regression cost function (log loss)

$$
J(\theta)=-\frac{1}{m}\sum^m_{i=1}{[y^ilog(\hat{p}^i)+(1-y^i)log(1-\hat{p}^i)]}
$$

> This cost function is **convex**, so **Gradient Descent** (or any other **optimization algorithm**) is guaranteed to find the **global minimum**.

- Logistic cost function partial derivatives

$$
\frac{\partial}{\partial\theta_j}J(\theta)=\frac{1}{m}\sum^m_{i-1}{(\sigma}(\theta^Tx^i)-y^i)x^i_j
$$

### Code

```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression() # param C:The higher the value of C, the less the model is regularized. (defaulf l2 penalty)
log_reg.fit(X, y)
```

## Softmax Regression (Multinomial Logistic Regression)

### Model

- Softmax score for class k

$$
s_k(x)=x^T\theta^{(k)}
$$

- Softmax function

$$
\hat{p}_k=\sigma(s(x))_k=\frac{exp(s_k(x))}{\sum_{j=1}^K{exp(s_j(x))}}
$$

> • K is the number of classes.
>
> • s(x) is a vector containing the scores of each class for the instance x.
>
> • σ(s(x)) k is the estimated probability that the instance x belongs to class k given the scores of each class for that instance.

Just like the Logistic Regression classifier, the Softmax Regression classifier predicts the class with the highest estimated probability (which is simply the class with the highest score).

### Cross entropy cost function

$$
J(Θ)=-\frac{1}{m}\sum^m_{i=1}\sum^K_{k=1}y^{(i)}_klog(\hat{p}^{(i)}_k)
$$

- parameter matrix Θ
- $y^{(i)}_k$ the target probability that the i th instance belongs to class k. In general, it is either equal to 1 or 0, depending on whether the instance belongs to the class or not.

### Code

```python
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
```



