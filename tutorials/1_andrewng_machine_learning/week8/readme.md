# Week 8: Unsupervised Learning and Dimenionality Reduction

## 8 Part 1: Unsupervised Learning

## Lecture 1: Clustering

### 1a: Unsupervised Learning: Introduction

* Finding structure in unlabelled data.

### 1b: K-means algorithm

* Inputs: unlabelled training dataset, number of classes (K)

* Iterative process: Cluster assignment step (for each training sample, compute $c^{(i)} = 1..K$, which stores the closest centroid.)  
                     Move centroid step (for each of the K centroids, compute $\mu^{(k)}$, which stores the mean n-vector).
                     
### 1c: Optimization objective

