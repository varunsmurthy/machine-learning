# Week 5: Neural Networks (Learning)

## Lecture 1: Cost Function and Back Propagation

### 1a: Cost Function

* Let's first define a few variables that we will need to use:

a) $L$ = total number of layers in the network  
b) $s_l$ = number of units (not counting bias unit) in layer l  
c) $K$ = number of output units or classes  

Recall that in neural networks, we may have many output nodes. We denote $h_\Theta(x)_k$ as being a hypothesis that results in the $k^{th}$ output. 

* For neural networks, the cost function is

$$\begin{equation*}
J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y^i_k log(h^i_k) + (1-y^i_k)\times(1-log(h^i_k)) + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_{l+1}} \sum_{j=1}^{s_l} ( \Theta_{i,j}^{(l)})^2  \end{equation*}$$

### 1b: Back-propagation algorithm

* "Backpropagation" is neural-network terminology for minimizing our cost function, i.e., $\min_\Theta J(\Theta)$.

* We need to compute the partial derivatives of $J(\Theta)$: $\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)$.