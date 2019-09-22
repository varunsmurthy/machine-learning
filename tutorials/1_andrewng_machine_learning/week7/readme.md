# Week 7: Support Vector Machines (SVMs)

## Lecture 1: Large Margin Classification

### 1a: Optimization Objective

* Replace the log function in the cost function for logistic regression with 2 piecewise linear cost functions for the y=1 ($Cost_1(X\theta)$) and y=0 ($Cost_0\theta)$) terms.  

* $Cost_1(X\theta)$ is 0 for $X\theta > 1$ and decreases linearly to 0 for increasing $X\theta$ from large negative values. Similarly, $Cost_0(X\theta)$ is 0 for $X\theta < -1$ increases linearly for greater values.

* Also multiply the cost function by $m$ and replace $\frac{1}{\lambda}$ by C. (see lecture notes for formulae).

* Finally, the SVM only ouputs discrete values of 0 and 1 instead of continuously varying values like logistic regression (see lecture notes for formulae).

### 1b: Large margin intuition

* Consider the previously discussed costfunction which contains two terms: The sum of cost1 and cost0 functions for each training set, multiplied by C, and the regularization term without $\lambda$ (note that C is $\frac{1}{\lambda}$).

* If C is very large, for the cost function to be minimized, the first term (sum of cost1 and cost0) needs to be very small. For this to happen, cost1 and cost2 should be close to 0, or, $\theta x^{(i)} > 1$ for y=1 and $\theta x^{(i)} < -1$ for y=0. This condition is more stringent compared to the logistic regression requirement of being greater or lesser than 0. Thus, SVM classification is very apt for classification for datasets with large margin. 

* However, if C is very large, this leads to overfiting or high variance issues. Hence optimum values for C need to be found.

### 1c: Mathematics behind large margin classification

* Relationship between vector inner product and projection of vectors (explained using 2-D vectors and norm). Inner product of two vectors is equal to the product of the magnitude of the projection and the norm of the vector on which the other vector is projected.

* Consider the optimization objective for SVMs (the cost function) and assume that C is very large. Rewriting the regularization term using the norm of the $\theta$ vector, we get,

$$
\begin{align*} \min\_\theta & \frac{1}{2}\sum\_{j=1}^n\theta\_j^2\\ \mbox{s.t.} & \|\theta\|\cdot p^{(i)} \geq 1\quad \mbox{if}\ y^{(i)} = 1\\ & \|\theta\|\cdot p^{(i)} \leq -1\quad \mbox{if}\ y^{(i)} = 0 \end{align*}
$$

* Consider $\theta\_0 = 0$ so that the decision boundary passes through the origin to simplify things. The $\theta$ vector is perpendicular to the decision boundary. Thus, the decision boundary is chosen with a large margin so as to maximize (or minimize) $p^{(i)}$.