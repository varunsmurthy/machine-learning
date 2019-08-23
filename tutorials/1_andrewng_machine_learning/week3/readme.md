# Week 3: Logistic Regression and Regularization

## Lecture 1: Classification and Representation

### 1a: Classification

* Two-class or binary-class vs multi-class classification problem.

* One approach could be to use linear regression on the training data set and subsequently defining a threshold for classification. However, this method doesn't work well because classification is not actually a linear function (i.e., an outlier might severely change the slope of the linear regression, resulting in classification errors).

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

$$ \mathrm{Cost}(h_\theta(x),y) = - y \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$$

* The vectorized form of the cost equation is

\begin{align*} & h = g(X\theta)\newline & J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \end{align*}

* The gradient descent algorithm can be used to minimize the cost function for Logistic regression and has the same form as the gradient descent algorithm for linear regression:

\begin{align*} & Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}

* The vector form of the gradient descent algorithm is 

$$ \theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y}) $$


### 2c: Advanced optimization

* "Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent.

* We first need to provide a function that evaluates the cost function and the various partial derivatives of the cost function with respect to the model parameters.

* Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()".

## Lecture 3: Multiclass Classification

### 3a: Multiclass Classification: One-vs-all

* In multi-class classification, y can take multiple values. Say, y = {0,1...n}.

* We divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

* $$ \begin{align*}& y \in \lbrace0, 1 ... n\rbrace \newline& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline& \cdots \newline& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline\end{align*} $$

## Lecture 4: Solving the problem of overfitting

### 4a: Problem of overfitting

* Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

* There are two main options to address the issue of overfitting:

    * Reduce the number of features:

        * Manually select which features to keep.
        * Use a model selection algorithm (studied later in the course).
    * Regularization

        * Keep all the features, but reduce the magnitude of parameters $θ_j$.
        * Regularization works well when we have a lot of slightly useful features.
        
### 4b: Cost function

* Regularization: If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

* $$ \begin{equation}
J_{\theta} = \frac{1}{2m} \sum\limits_{i=1}^m (h(x^{(i)}) - y^{(i)})^2 + \lambda \sum\limits_{j=1}^n \theta_j^2
\end{equation} $$

* The λ, or lambda, is the regularization parameter. It determines how much the costs of our theta parameters are inflated.