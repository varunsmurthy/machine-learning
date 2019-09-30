# Week 8: Unsupervised Learning and Dimensionality Reduction

## 8 Part 1: Unsupervised Learning

## Lecture 1: Clustering

### 1a: Unsupervised Learning: Introduction

* Finding structure in unlabelled data.

### 1b: K-means algorithm

* Inputs: unlabelled training dataset, number of classes (K)

* Iterative process: Cluster assignment step (for each training sample, compute $c^{(i)} = 1..K$, which stores the closest centroid.)  
                     Move centroid step (for each of the K centroids, compute $\mu^{(k)}$, which stores the mean n-vector).
                     
### 1c: Optimization objective

* Minimize the cost function, which is the sum of the squares of the distance of each training set sample and its assigned centroid (see notes for formula).

### 1d: Random initialization

* To pick the initial K centroids (where K is the number of classes), pick K random integers from i..m.

* If K is small, then repeat the clustering for a few 100 times and choose the classification with the lowest cost function.

### 1e: Choosing the number of clusters

* Use the elbow method (plot the cost function on the y axis as a function of the number of clusters; then pick the pint at which the cost function no longer rduces drastically).


## 8 Part 2: Dimensionality Reduction

## Lecture 2: Motivation for dimensionality reduction

### 2a: Motivation 1: Data compression

* If any two features are well correlated, they can be clubbed to form one feature. Similarly, if three features all predominantly lie in one plane, they can be clubbed to form two features. This is useful to reduce the memory taken by data as well for models to run faster.

### 2b: Motivation 2: Data Visualization

* Reducing the number of dimensions could also make it easier to visualize data (e.g., reducing 50 features into 2-3 features makes it easier to visualize the datset).

## Lecture 3: Principal Component Analysis (PCA)

### 3a: PCA problem formulation

* Reduce from n-dimnsions to k-dimensions: Find k vectors $u^{(1)}...u^{(k)}$ onto which to project each training set sample, so as to minimize the projection error.

### 3b: PCA algorithm

* Perform feature scaling and mean normalization.
* Compute the covariance matrix of the training sample ($\Sigma$). Check notes for formula; vectorized implementation is $\frac{X^T X}{m}$.
* Compute the eigen vector of $\Sigma$; the first k columns are the k $n \times 1$ vectors which form the plane onto which the training samples are projected on.
* $z^{(i)}$ is $u^T x^{(i)}$ and is a $k \times 1$ vector.

## Lecture 4: Applying PCA

### 4a: Reconstruction from compressed representation

* Obtaining $x^{(i)}$ from $z^{(i)}$: $x\_{approx}^{(i)} = u z^{(i)}$.

### 4b: Choosing the number of principal components

* Define the average squared projection error: average squared distance between $x^{(i)}$ and $x\_{approx}^{(i)}$.  
* Define the variance in the data: average squared norm of the data.  
* The number of principal components is the smallest value $k$ for which the ratio of the average squared projection error and the total variance in data is less than 0.01 or 1%. In other words, the smallest number of pricipal components for which 99% of the variance in data is retained.
* One approach is to check the variance captured for different values of $k$ and chose the value for which 99% of the variance is captured. 
* Alternatively, the matrix $S$ that is given by $[U,S,V] = svd(\Sigma)$ is a $n \times n$ diagonal matrix whose diagonal elements $S\_{ii}$ gives the cumulative variance captured by the first $i$ eigen vectors. 