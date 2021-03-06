# Week 9: Anomaly Detection and Recommender Systems

## 9 Part 1: Anomaly Detection

## Lecture 1: Density Estimation

### 1a: Problem Motivation

* Detect anomalous features: faulty product in manufactoring, fraudulent user on website, disfunctional computer in datacenter.  
* Model representation: if $p(x) > \epsilon$, not anomalous, if $p(x) < \epsilon$, anomalous.

### 1b: Gaussian Distribution

* Gaussian or normal distribution (as a function of mean and variance): See notes for formula

### 1c: Anomaly detection algorithm

* The aim is to compute the prbability distribution for a variable with $n$ features. Assuming that the features are mutually independent, the probability distribution of the $x$ is the product of the individual Normal distributions of the $n$ features. So, the means and standard deviations have to be computed for all the features.

## Lecture 2: Building an Anomaly Detection system

### 2a: Developing and Evaluating an Anomaly Detection System

* Evaluating an anomaly detection system requires a quantitative score. For this, we treat the problem as a supervised learning problem, with labels available.

* The training data is split into training data (60%, with no or very few anomalous samples), cross-validation data (20%, with about half the anomalous samples), and the test data (20%, with about half the anomalous samples). The training set is used to compute the mean and variance of features, and the cross validation set is used to fix the value of $\epsilon$ used for predicting anomalous samples.

* Since the data is skewed, anomaly detection is evaluated using precision, recall, or the F1 score. $\epsilon$ is chosen to maximize these quantities in the cross-validation data.

### 2b: Anomaly Detection vs. Supervised Learning

* Use anomaly detection when there are a large number of non-anomalous samples and a few anomalous samples. Use supervised learning when we have large number of positive and negative samples.

* Use anomaly detection when there are many types of anomalies, which may or may not be known. For e.g., airplane faults may not be known, thus it is better to predict anomalous samples as those that are distant from non-anomalous samples.

### 2c: Choosing What Features to Use

* Sometimes, the features might not follow a Gaussian distribution. Transformations can be applied to make the feature appear closer to a Gaussian distribution (log, exponents, square root, etc.).

* Also, $p(x)$ for an anomalous sample might be similar to those of non-anomalous samples. In such cases, try to arrive at new features which might differentiate the anomalous sample from the others.

## Lecture 3: Multi-variate Gaussian Distribution

### 3a: Multivariate Gaussian Distribution

* If there are any correlations between features, then the product of the univariate gaussian distributions of the features may not function well. In such cases, we can use the multi-variate Gaussian distribution. Here, mean $\mu \in R^n$ and the covariance matrix $\Sigma \in R^{n \times n}$. For formula of the distribution, see notes. 

* The diagonal elements of the covariance matrix $\Sigma$ represent the variance in the different features and the off diagonal elements represent the correlation between the features (positive values imply positive correlation, larger values imply higher correlation, i.e., narrower scatter).

### 3b: Anomaly Detection using the Multivariate Gaussian Distribution

* The multivariate Gaussian model can automatically capture correlations between different features in x. However, the multivariate model is computationally extensive when $n$ is very large (since the inverse of the covariance matrix $\Sigma$ has to be computed). Furthermore, $\Sigma$ is non-invertible when $m < n$, and if there are any features that are linearly dependent (redundant).

* The univariate model $p(x\_1;\mu\_1,\sigma\_1^2)\times\dots\times p(x\_n;\mu\_n,\sigma\_n^2)$ corresponds to a multivariate Gaussian where the contours of $p(x;\mu,\Sigma)$ are axis-aligned. However, since this model cannot explicitly capture the correlations between features, these relations between features have to be engineered (by adding additional features).

## 9 Part 2: Recommender Systems

## Lecture 4: Predicting Movie Ratings

### 4a: Problem Formulation

* $n\_u, n\_m, r(i,j), y^{(i,j)}$: number of users, number of movies, 1 if the $i^{th}$ movie has been rated by the $j^{th}$ user, value of the rating.

* The problem statement of the movie recommendation system is to predict $y^{(i,j)}$ when $r(i,j)$ is undefined.

### 4b: Content based recommendations

* Model features could possibly define some content-based characteristics of the movies (e.g., action, romance, etc.) as a quantitative value (a funny action movie could have features as comedy = 0.5 and action = 0.5). For each user $j$, we can then form a hypothesis for movie rating prediction based on linear regression. The model parameters for each user will be $\theta^{(j)}$.

* Linear regression can be applied by first noting the cost function (for all users) and minimizing it using gradient descent (see notes for notes).

* However, this method of predicting movie ratings for all users requires information about the content of each movie, which may not be possible to obtain for a large number of movies.

## Lecture 5: Collaberative Filtering

### 5a: Collaberative Filtering

* Here, let's say we have the model parameters for all the users (i.e., $\theta^{(j)}$ for all users) and some movie ratings $y^{(i,j)}$ just as in the case of content based recommendations.

* The goal is to find the model features for all the movies (i.e., $x^{(i)}$ for all movies). We can now write a cost function for each movie with the cost function being dependent on $x^{(i)}$ instead of $\theta^{(j)}$. The regularization term also has $x^{(i)}$ instead of $\theta^{(j)}$.

### 5b: Collaberative filtering algorithm

* We can alternatively apply content based recommendation and collaberative filtering to learn the model features for each movie and the model parameters for each user.

* Alternatively, we can write a cost function that is dependent on $x^{(i)}$ and $\theta^{(j)}$ and minimize it with respect to both.