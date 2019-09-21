# Week 7: Support Vector Machines (SVMs)

## Lecture 1: Large Margin Classification

### 1a: Optimization Objective

* Replace the log function in the cost function for logistic regression with 2 piecewise linear cost functions for the y=1 ($Cost_1(X\theta)$) and y=0 ($Cost_0\theta)$) terms.  

* $Cost_1(X\theta)$ is 0 for $X\theta > 1$ and decreases linearly to 0 for increasing $X\theta$ from large negative values. Similarly, $Cost_0(X\theta)$ is 0 for $X\theta < -1$ increases linearly for greater values.

* Also multiply the cost function by $m$ and replace $\frac{1}{\lambda}$ by C. (see lecture notes for formulae).

* Finally, the SVM only ouputs discrete values of 0 and 1 instead of continuously varying values like logistic regression (see lecture notes for formulae).

### 1b: Large margin intuition

* Consider the previously discussed costfunction which contains two terms: The sum of cost1 and cost0 functions for each training set, multiplied by C, and the regilarization term without $\lambda$ (note that C is $\frac{1}{\lambda}$).

* If C is very large, for the cost function to be minimized, the first term (sum of cost1 and cost0) needs to be very small. For this to happen, cost1 and cost2 should be close to 0, or, $\theta x^{(i)} > 1$ for y=1 and $\theta x^{(i)} < -1$ for y=0. This condition is more stringent from the logistic regression requirement of being greater or lesser than 0. Thus, SVM classification is very apt for classification for datasets with large margin. 

* However, if C is very large, this leads to overfiting or high variance issues. Hence optimum values for C need to be found.