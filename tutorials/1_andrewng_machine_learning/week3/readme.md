# Week 3: Logistic Regression and Regularization

## Lecture 1: Classification and Representation

### 1a: Classification

* Two-class or binary-class vs multi-class classification problem.

* One approach could be to use linear regression on the training data set and subsequently defining a threshold for classification. However, this method doesn't work well because classification is not actually a linear function.

### 1b: Hypothesis representation

* Let the form for our hypotheses $h_\theta(x)$ to satisfy $0 \leq h_\theta(x) \leq 1$. This is accomplished by plugging $h_\theta(x) = \theta^T X$ into the Logistic Function (also called the Sigmoid function).

* \begin{align*}& h_\theta (x) = g ( \theta^T x ) \newline \newline& z = \theta^T x \newline& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*}

* The function $g(z)$, shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

* $h_\theta(x)$ will give us the probability that our output is 1. For example, $h_\theta(x) = 0.7$ gives us a probability of 70% that our output is 1.

### 1c: Decision boundary

* This is the curve in $x$ space for which $\theta^T x = 0$ or $h_\theta (x) = 0.5$, i.e., there is equal probability of $y=0$ and $y=1$.

* $\theta^T x > 0$ indicates regions where probability of $y=1$ is greater, and $\theta^T x < 0$ indicates regions where probability of $y=0$ is greater.

## Lecture 2: Logistic Regression Model

### 2a: Cost function

* We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

* Instead, our cost function for logistic regression looks like:

\begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}

* Here, the cost function is zero if the model prediction is accurate and tends to infinity in the case of a false negative ($h_\theta(x)=0$ when $y=1$) or a false positive ($h_\theta(x)=1$ when $y=0$).

\begin{align*}& \mathrm{Cost}(h_\theta(x),y) = 0 \text{ if } h_\theta(x) = y \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 \newline \end{align*}

### 2b: Simplified cost function and gradient descent

* The conditional cost function can be combined as 

$$ \mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$$

* The vectorized form of the cost equation is

\begin{align*} & h = g(X\theta)\newline & J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \end{align*}