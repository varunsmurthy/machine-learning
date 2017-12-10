# Week 3: Logistic Regression and Regularization

## Lecture 1: Classification and Representation

### 1a: Classification

* Two-class or binary-class vs multi-class classification problem.

* One approach could be to use linear regression on the training data set and subsequently defining a threshold for classification. However, this method doesn't work well because classification is not actually a linear function.

### 1b: Hypothesis representation

* Let the form for our hypotheses $h_\theta(x)$ to satisfy $0 \leq h_\theta(x) \leq 1$. This is accomplished by plugging $h_\theta(x) = \theta^T X$ into the Logistic Function.

* \begin{align*}& h_\theta (x) = g ( \theta^T x ) \newline \newline& z = \theta^T x \newline& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*}

* The function $g(z)$, shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

* $h_\theta(x)$ will give us the probability that our output is 1. For example, $h_\theta(x) = 0.7$ gives us a probability of 70% that our output is 1.

### 1c: Decision boundary