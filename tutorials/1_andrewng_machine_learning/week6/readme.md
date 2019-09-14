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