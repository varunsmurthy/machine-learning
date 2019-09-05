# Week 5: Neural Networks (Learning)

## Lecture 1: Cost Function and Back Propagation

### 1a: Cost Function

* Let's first define a few variables that we will need to use:

a) $L$ = total number of layers in the network  
b) $s_l$ = number of units (not counting bias unit) in layer l  
c) $K$ = number of output units or classes  

Recall that in neural networks, we may have many output nodes. We denote $h_\Theta(x)_k$ as being a hypothesis that results in the $k^{th}$ output. 

* For neural networks, the cost function is

$$\begin{align*}
J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y^{(i)}_k \log(h(x^{(i)})_k)  \end{align*}$$

