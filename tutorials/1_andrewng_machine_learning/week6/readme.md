# Week 6: Advice for Applying ML and ML System Design

## 6-Part I: Advice for Applying ML

## Lecture 1: Evaluating a Learning Algorithm

### 1a: Deciding what to try next

* If the model does not work well with input data, different things can be tried (e.g., add more features, add non-linear features, reduce number of features, modify regularization parameter, etc.). Certain diagnostics can be used to get more insight into future steps to improve the model.

### 1b: Evaluating a hypothesis

* A hypothesis may have a low error for the training examples but still be inaccurate (because of overfitting). Thus, to evaluate a hypothesis, given a dataset of training examples, we can split up the data into two sets: a training set and a test set. Typically, the training set consists of 70 % of your data and the test set is the remaining 30 %.

* The test set error can just be the cost function computed for the test data set with the computed hypothesis. For logistic regression, we can also compute a misclassification error, which is 1 for a false positive or negative and 0 otherwise. This misclassification error is summed and averaged over the whole test data set to get the fraction of test data points classified incorrectly.

### 1c: Model selection and training/validation/test sets 

* How do we chose the best model with regards to the polynomial degree (e.g., linear vs quadratic vs cubic)? Given many models with different polynomial degrees, we can use a systematic approach to identify the 'best' function. In order to choose the model of your hypothesis, you can compute the error for each degree of polynomial and look at the error result. To do this we split the training dataset into a training set (60%), a cross-validation set (20%), and a test set (20%).

* We can now calculate three separate error values for the three different sets using the following method:
    1) Optimize the parameters in $\Theta$ using the training set for each polynomial degree.
    2) Find the polynomial degree $d$ with the least error using the cross validation set.
    3) Estimate the generalization error using the test set with $J\_{test}(\Theta^{(d)})$, ($d$ = theta from polynomial with lower error);
    
## Lecture 2: Bias and Variance    

### 2a: Diagnosing bias vs variance

* In this section we examine the relationship between the degree of the polynomial d and the underfitting or overfitting of our hypothesis. High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two. We need to distinguish whether bias or variance is the problem contributing to bad predictions.

* The training error will tend to decrease as we increase the degree d of the polynomial. At the same time, the cross validation error will tend to decrease as we increase d up to a point, and then it will increase as d is increased, forming a convex curve.

* High bias (underfitting): $J\_{train}(\Theta^{(d)})$ is high, $J\_{cv}(\Theta^{(d)})$ is high  
  High variance (overfitting): $J\_{train}(\Theta^{(d)})$ is low, $J\_{cv}(\Theta^{(d)})$ is high
  
### 2b: Regularization and Bias/Variance

* Regularization is a process by which the model parameters are scaled by some value $\lambda$ to prevent overfitting. As a result, when $\lambda$ is small, we have overfitting or high variance, and when $\lambda$ is large, we have underfitting or high bias. So, how do we choose the right $\lambda$? The following procedure is used:

    1) Create a list of lambdas (i.e. 0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24).
    2) Create a set of models with different degrees or any other variants.
    3) Compute the cost function (without regularization terms) on the cross validation set for all $\lambda$s and all degrees.
    4) Select the best combo that produces the lowest error on the cross validation set, and use this combination on the test set.
    
* Variation of cost function with $\lambda$: 
    1) High $\lambda$: underfitting (high bias), $J\_{train}(\Theta)$ is high, $J\_{cv}(\Theta)$ is high .
    2) Low $\lambda$: overfitting (high variance), $J\_{train}(\Theta)$ is low, $J\_{cv}(\Theta)$ is high .
    
### 2c: Learning curves

* Plotting the cost function for the training ($J\_{train}(\Theta)$) and cross validation ($J\_{cv}(\Theta)$) sets as a function of the training set size ($m$). $J\_{train}(\Theta)$ will typically increase with $m$ as it becomes harder for the model to fit each sample. $J\_{cv}(\Theta)$ will typically decrease with $m$ as the model generally becomes better when it is being trained on a larger number of samples.

* High bias case (underfitting): 
    1) $J\_{train}(\Theta)$ quickly increases to a large value as $m$ increases since the model is unable to capture the variance in the training samples.
    2) $J\_{cv}(\Theta)$ initially decreases as the model improves with larger $m$, but quickly plateaus to a large value.
    3) Since $J\_{cv}(\Theta)$ plateaus to a high value, increasing the size of the training set by collecting more data is not helpful.
    4) For a high bias case, the characteristic feature of the learning curves is the high values of both $J\_{train}(\Theta)$ and $J\_{cv}(\Theta)$ and the small difference between them.
    
* High variance case (overfitting):
    1) $J\_{train}(\Theta)$ slowly increases as $m$ increases since the model is unable to capture all the variance in the training samples.
    2) $J\_{cv}(\Theta)$ decreases as the model improves with larger $m$ and gets closer to $J\_{train}(\Theta)$.
    3) Thus, increasing the size of the training set is helpful in a high variance model.
    4) For a high variance case, the characteristic feature of the learning curves is the gradually decreasing difference between $J\_{train}(\Theta)$ and $J\_{cv}(\Theta)$ as $m$ increases.
    

### 2d: Deciding what to do next revisited

* Our decision process can be broken down as follows:
    1) Getting more training examples: Fixes high variance
    2) Trying smaller sets of features: Fixes high variance
    3) Adding features: Fixes high bias
    4) Adding polynomial features: Fixes high bias
    5) Decreasing λ: Fixes high bias
    6) Increasing λ: Fixes high variance.
    
* Diagnosing Neural Networks: 
    1) A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper.
    2)A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase $\lambda$) to address the overfitting.

* Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.


## 6-Part II: ML System Design

## Lecture 3: Building a Spam classifier

### 3a: Prioritizing What to Work On

* System design example: Spam classifier
    1) Decide on model features  (absence or presence of some $n$ separate words; 0s and 1s)
    2) Check if high bias or high variance and perform relevant correction techniques.
    
### 3b: Error analysis

* Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.

* It is very important to get error results as a single, numerical value. Otherwise it is difficult to assess your algorithm's performance.

## Lecture 4: Handling Skewed Data

### 4a: Error metrics for skewed classes

* Consider a binary classification problem where one of the classes occurs very rarely (e.g., malignant tumour in a general population). In such a case, we use two error metrics 

    1) Precision: Fraction of all positive predictions that were correct (= true +ve/(true +ve + false +ve))
    2) Recall: Fraction of all actual positive occurrences that were correctly identified (= true +ve/(true +ve + false -ve))
    
### 4b: Trading Off Precision and Recall

* Instead of using 0.5 as the threshold for $h\_\theta(x)$ in logistic regression, use a varying threshold and compute the F score and maximize it to get the best combined values of precision and recall.  

$$
F\_1 = 2\frac{PR}{P+R}
$$