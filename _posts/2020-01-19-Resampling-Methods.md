---
title: "Resampling Methods"
date: 2020-01-19
categories: Machine-Learning
---

[TOC]

# Introduction

Resampling methods involve repeatedly drawing samples from a training set and reﬁtting a model of interest on each sample in order to obtain additional information about the ﬁtted model. (e.g. cross-validation, bootstrap)

- Estimates of test-set prediction error (CV)
- S.E. and bias of estimated parameters (Bootstrap)
- C.I. of target parameter (Bootstrap)

# Cross-Validation

The **training error rate** often is quite diﬀerent from the **test error rate**, and in particular the former can dramatically underestimate the latter.

- Model Complexity Low: High bias, Low variance

- Model Complexity High: Low bias, High variance

**Prediction Error Estimates**

- Large test set
- Mathematical adjustment
  - $C_p=\frac{1}{n}(SSE_d+2d\hat{\sigma}^2)$
  - $AIC=\frac{1}{n\hat{\sigma}^2}(SSE_d+2d\hat{\sigma}^2)$
  - $BIC=\frac{1}{n\hat{\sigma}^2}(SSE_d+log(n)d\hat{\sigma}^2)$
- CV: Consider a class of methods that estimate the test error rate by <u>holding out a subset of the training observations</u> from the ﬁtting process, and then <u>applying the statistical learning method to those held out observations</u>.

## The Validation Set Approach

A random splitting into two halves: left part is training set, right part is validation set.

### Drawbacks

- The validation estimate of the test error rate can be **highly variable**, depending on precisely which observations are included in the training set and which observations are included in the validation set.
- Only a subset of the observations are used to ﬁt the model.
- Validation set error rate may tend to overestimate the test error rate for the model ﬁt on the entire data set.

## Leave-One-Out Cross-Validation

LOOCV involves splitting the set of observations into two parts. However, instead of creating two subsets of comparable size, a single observation $(x_1 , y_1 )$ is used for the validation set, and the remaining observations ${ (x_2 , y_2 ), . . . , (x_n , y_n ) }$ make up the training set.

### In Linear Regression

$$
CV_{(n)}=\frac{1}{n}\sum^n_{i=1}(\frac{y_i-\hat{y_i}}{1-h_i})^2
$$

- $CV_n$ bacomes a weighted MSE

### Drawbacks

- Estimates from each fold are **highly correlated** and hence their average can have **high variance**.

## K-fold Cross-Validation

This approach involves randomly dividing the set of observations into k groups, or folds, of approximately equal size. The ﬁrst fold is treated as a validation set, and the method is ﬁt on the remaining k − 1 folds. This procedure is repeated k times; each time, a diﬀerent group of observations is treated as a validation set. This process results in k estimates of the test error. The k-fold CV estimate is computed by averaging these values. If k=n, then it is LOOCV.
$$
CV_{(k)}=\frac{1}{k}\sum^k_{i=1}{MSE}\\or\\
CV_{(k)}=\frac{1}{k}\sum^k_{i=1}{Err_k}
$$
Typically, given these considerations, one performs k-fold cross-validation using k = 5 or k = 10, as these values have been shown empirically to yield test error rate estimates that suﬀer neither from excessively high bias nor from very high variance.

# Bootstrap

- A powerful statistical tool to **quantify the uncertainty** associated with a given estimator or statistical learning method.
- For example, it can provide an estimate of the **standard error** of a coeﬃcient, or a **conﬁdence interval** for that coeﬃcient.

### Steps

- Obtain datasets ($n$ observations) by repeatedly sampling from the **original data set** $Z$ with **replacement** $B$ times.

- Each of these **bootstrap data**, denoted as $Z^{*1},..., Z^{*B}$, is the **same size** as original dataset $n$. And bootstrap estimates for $\alpha$ denoted as $\hat{\alpha}^{*1},..., \hat{\alpha}^{*B}$. Thus some observations may appear more than once and some not at all ([2/3 of original dataset](https://stats.stackexchange.com/questions/88980/why-on-average-does-each-bootstrap-sample-contain-roughly-two-thirds-of-observat)).

### Estimate of S.E.

$$
SE_B(\hat{\theta})=\sqrt{\frac{1}{B-1}\sum^B_{r=1}(\hat{\theta}^{*r}-\bar{\theta}^*)^2}
$$

### Estimate of C.I.

#### Bootstrap Percentile C.I.

$$
[L,U]=[\hat{\theta}^*_{\alpha/2}, \hat{\theta}^*_{1-\alpha/2}]
$$

#### Bootstrap S.E. based C.I.

$$
[L,U]=\bar{\theta}\pm z_{1-\alpha/2}\times\frac{SE^*}{B}
$$

#### Better Option (Basic Bootstrap/Reverse Percentile Interval)

$$
[L,U]=[2\hat{\theta}-\theta^*_{1-\alpha/2}, 2\hat{\theta}-\theta^*_{\alpha/2}]
$$

Key: the behavior of $\hat{\theta}^*-\hat{\theta}$ is approximately the same as the behavior of $\hat{\theta}-\theta$.

Therefore:
$$
\begin{align}
0.95 &\approx P(\hat{\theta}^*_{\alpha/2}\le\hat{\theta}^*\le\hat{\theta}^*_{1-\alpha/2}) \\
&= P(\hat{\theta}^*_{\alpha/2}-\hat{\theta}\le\hat{\theta}^*-\hat{\theta}\le\hat{\theta}^*_{1-\alpha/2}-\hat{\theta}) \\
& = P(\hat{\theta}^*_{\alpha/2}-\hat{\theta}\le\hat{\theta}^*-\theta\le\hat{\theta}^*_{1-\alpha/2}-\hat{\theta}) \\
&\approx P(\hat{\theta}^*_{\alpha/2}-\hat{\theta}\le\hat{\theta}-\theta\le\hat{\theta}^*_{1-\alpha/2}-\hat{\theta}) \\
& = P(2\hat{\theta}-\theta^*_{1-\alpha/2}\le\theta\le2\hat{\theta}-\theta^*_{\alpha/2})

\end{align}
$$

### In General

- Each bootstrap sample has signiﬁcant overlap with the original data. This will cause the bootstrap to seriously **underestimate** the true prediction error.
  - Can partly ﬁx this problem by only using predictions for those observations that did not ( by chance ) occur in the current bootstrap sample. (Complicated)

- If the data is a **time series**, we can’t simply sample the observations with replacement. We can instead create blocks of consecutive observations, and samp le those with replacements. Then we paste to gether sampled blocks to obtain a bootstrap samples.

## Bootstrap in Regression

$$
Y_i=\beta_0+\beta_1X_i+\epsilon_i,\ i=1,...,n
$$

Find S.E. and C.I. for $\beta_0$ and $\beta_1$

### Empirical Bootstrap

- Resampling $(X_1, Y_1), ..., (X_n, Y_n)$ and obtain:
  - Bootstrap sample 1: $(X_1^{*1}, Y_1^{*1}), ..., (X_n^{*1}, Y_n^{*1})$
  - Bootstrap sample 2: $(X_1^{*2}, Y_1^{*2}), ..., (X_n^{*2}, Y_n^{*2})$
  - ...
  - Bootstrap sample 1: $(X_1^{*B}, Y_1^{*B}), ..., (X_n^{*B}, Y_n^{*B})$

- For each Bootstrap sample, fit regression and obtain $(\hat{\beta}_0^{*1},\hat{\beta}_1^{*1})...(\hat{\beta}_0^{*B},\hat{\beta}_1^{*B})$, then estimate S.E. and C.I.

### Residual Bootstrap

- Recall that residuals to mimic the role of $\epsilon$.
- Bootstrap the residuals and obtain:
  - Bootstrap residual 1: $\hat{e}_1^{*1},...,\hat{e}_n^{*1}$
  - Bootstrap residual 1: $\hat{e}_1^{*2},...,\hat{e}_n^{*2}$
  - ...
  - Bootstrap residual 1: $\hat{e}_1^{*B},...,\hat{e}_n^{*B}$

- Generate new bootstrap sample: $X_i^{*b}=X_i,\ Y_i^{*b}=\hat{\beta}_0+\hat{\beta}_1X_i+\hat{e}_i^{*b}$

- For each bootstrap sample, fit regression and estimate S.E. and C.I. 

### Wild Bootstrap

When variance of error $Var(\epsilon_i|X_i)$ depends on the value of $X_i$ ( so called **heteroskedasticity**) , **residual bootstrap is unstable** because the residual bootstrap will swap all the residuals regardless of the value of X. But wild bootstrap uses the residual of itself only.

- Generate IID random variables $V_1^b,...,V_n^b \sim N(0,1)$

- Generate new bootstrap sample: $X_i^{*b}=X_i,\ Y_i^{*b}=\hat{\beta}_0+\hat{\beta}_1X_i+V_i^b\hat{e}_i$

- For each bootstrap sample, fit regression and estimate S.E. and C.I. 