# Week 2: Linear Regression with Multiple Variables

## Lecture 1: Multivariate Linear Regression

### 1a: Multiple features

* \begin{equation*} x_j^{(i)} = \text{value of feature } j \text{ in the }i^{th}\text{ training example} \end{equation*}

* The multivariable form of the hypothesis function accommodating these multiple features is as follows:
 \begin{equation*}h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n\end{equation*}

\begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x\end{align*}

### 1b: Gradient descent for multiple variables

\begin{align*}& \text{repeat until convergence:} \lbrace \newline & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\newline \rbrace\end{align*}

### 1c: Gradient descent in practice 1: Feature scaling

* Modify the ranges of our input variables so that they are all roughly the same.

* Two techniques to help with this are feature scaling and mean normalization.

* $x_i := \dfrac{x_i - \mu_i}{s_i}$ where $s_i$ could be the range or the standard deviation.

### 1d: Gradient descent in practice 2: Learning rate

* Debugging gradient descent. Make a plot with number of iterations on the x-axis. Now plot the cost function, $J(\theta)$ over the number of iterations of gradient descent. If $J(\theta)$ ever increases, then you probably need to decrease $\alpha$.

### 1e: Features and polynomial regression

* Multiple features can be combined into one.

* For a polynomial regression, set the higher powers of the independent variable as various model parameters ($x_2 = x_1^2$, $x_3 = x_1^3$).

## Lecture 2: Computing Parameters Analytically

### 2a: Normal equation

* Minimizing the cost function $J(\theta)$, i.e., determining the $(n+1)$ vector $\theta$, used the gradient descent approach. Alternatively, the normal equation method involves setting the partial derivatives of the cost function with respect to the various model parameters $(i.e., \theta_0, \theta_0, ..)$ to zero. This gives us $(n+1)$ equations for the $(n+1)$ unknown $\theta$ variables. Solving these,

$$ \theta = (X^T X)^{-1}X^T y $$  

where $X$ is called the design matrix, and each row of the design matrix represents the inputs of the $i^{th}$ training data set (i.e., each row of $X$ is $x^{(i)T}$).

* No need to choose the learning parameter, and no need to iterate. However, slow for large values($> 10^6$) of the model parameters ($n$). Computing the inverse of $X^T X$ is an $O(n^3)$ operation.

### 2b: Normal equation non-invertibility

* If $X^T X$ is non-invertible, then the normal equation wouldn't have a solution. Common causes:
    * Redundant model variables.
    * Too many model variables ($n>m$).
    
* Using pinv instead of inv in Matlab (pseudo inverse vs inverse) will work in the case of non-ivertibility.

## Lecture 3: Octave/Matlab Tutorial

### 3a: Basic operations

### 3b: Moving data around

### 3c: Computing on data

### 3d: Plotting data

### 3e: Control statements: for, while, and if

### 3f: Vectorization

