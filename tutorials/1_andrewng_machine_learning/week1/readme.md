# Week 1: Introduction, linear regression with one variable, and linear algebra review

## Lecture 1: Introduction

### 1a: What is machine learning?

* Machine learning: idea of teaching a computer to learn concepts using data—without being explicitly programmed.

* Learning algorithms: A computer program is said to learn from experience E, with respect to some task T, and some performance measure P, if its performance on T as measured by P improves with experience E.

* Two types of learning algorithms: Supervised and unsupervised. Separately, also reinforcement and recommender systems.

### 1b: Supervised learning

* The term supervised learning refers to the fact that we gave the algorithm a data set in which the "right answers" were given.

* Regression problem (output being predicted is a continuous variable) vs classification problem (output being predicted is discrete).

### 1c: Unsupervised learning

* We're given data that doesn't have any labels or that all have the same label or really no labels.

* Clustering: Classifying the data into separate clusters.

## Lecture 2: Model and Cost Function

### 2a: Model representation

* To establish notation for future use, we’ll use $x^{(i)}$ to denote the “input” variables, also called input features, and $y^{(i)}$ to denote the “output” or target variable that we are trying to predict.

* A pair $(x^{(i)},y^{(i)})$ is called a training example, and the dataset that we’ll be using to learn —- a list of $m$ training examples $(x^{(i)},y^{(i)});i=1,...,m$ —- is called a training set.

* To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function $h : X \rightarrow Y$ so that $h(x)$ is a “good” predictor for the corresponding value of $y$. For historical reasons, this function h is called a hypothesis.

* If $h(x)$ is a linear function of $x$, then the learning model is a linear regression in one variable (also called an univariate linear regression).

### 2b: Cost function

* A univariate linear regression model is given by $h_{\theta}(x) = \theta_0 + \theta_1 x$, where $\theta_0$ and $\theta_1$ are called model parameters. The crux of the learning model is to determine $\theta_0$ and $\theta_1$, and this is done by minimizing a cost function $J_{\theta}$.

* An example of a cost function is the squared error cost function, given by 

\begin{equation}
J_{\theta} = \frac{1}{2m} \sum\limits_{i=1}^m (h(x^{(i)}) - y^{(i)})^2
\end{equation}

### 2c: Cost function - Intuition 1

* Considering a model $h_{\theta}(x)$ where $\theta_0 = 0$, the cost function $J = J(\theta_1)$. Plotting $J$ against $\theta_1$ gives us the value of $\theta_1$ for which $J$ is minimized, and hence the model $h_{\theta}(x)$.

### 2d: Cost function - Intuition 2

* If $\theta_0 != 0$ in $h_{\theta}(x)$, then $J = J(\theta_0, \theta_1)$. Hence the plot of the cost function, $J$, is a 2-D contour as a function of $\theta_0$ and $\theta_1$. The minima of $J$ gives the values of $\theta_0$ and $\theta_1$, and hence the model $h_{\theta}(x)$.

## Lecture 3: Parameter Learning

### 3a: Gradient descent

* Determining the model parameters, $\theta_0$ and $\theta_1$, depends on minimizing the cost function $J = J(\theta_0, \theta_1)$. One way to do this is by the method of gradient descent, which can be imagined by starting at an arbitrary point on the $J$ contour in $(\theta_0, \theta_1)$ space, and then moving sequentially in the direction of downward slope.

* The gradient descent algorithm is 

\begin{equation}
\text{Repeat until convergence:}\ \
\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}
\end{equation}

where $j$ is the id of the model parameter. In the case if a 1-D linear regression, $j$ ranges from 0 to 1. $\alpha$ is the learning rate, and denotes the step size while approaching the local minima.

* $\theta_j$ values should be updated simultaneously.

### 3b: Gradient descent intuition

* Explanation of how the algorithm works when $J = J(\theta_1)$.

### 3c: Gradient descent for linear regression

* We know the mean squared error cost function for the 1-D inear regression, and also the general gradient descent algorithm. Combining them, we get

\begin{align*} 
    \text{repeat until convergence: } \lbrace & \newline 
    \theta_0 := & \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(h_\theta(x_{i}) - y_{i}) \newline 
    \theta_1 := & \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}\left((h_\theta(x_{i}) - y_{i}) x_{i}\right) \newline 
    \rbrace& 
\end{align*}

* The cost function for the 1-D linear regression only has a global minima, and no local minima, so the gradient descent algorithm is quite appropriate.

* This particular implementation of the gradient descent algorithm makes use of all training dataset values at each iteration, and hence is called the "batch" gradient descent algorithm.

## Lecture 4: Linear Algebra Review

### 4a: Matrices and vectors

### 4b: Addition and scalar multiplication

### 4c: Matrix vector multiplication

### 4d: Matrix multiplication

### 4e: Matrix multiplication properties

### 4f: Inverse and transpose
