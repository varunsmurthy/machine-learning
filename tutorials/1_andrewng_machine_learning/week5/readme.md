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

* We need to compute the partial derivatives of $J(\Theta)$: $\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)$. To do this, we use the back-propagation algorithm.

* First, we define $\delta^{(l)}$, which is the error for each layer. $\delta^{(l)}$ is $s_l \times 1$ vector. For the output layer, $\delta^{(L)} = a^{(L)} - y$ for one training set sample $(x,y)$. Subsequently, 

$$ \delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)}) .* (a^{(l)} .* (1 - a^{(l)}) \hspace{4mm} for \hspace{4mm} l = L-1,....,2$$.

* Now that we have defined the error for each layer, the backpropagation algorithm given a training set $\lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace$ is as follows:

    A. For training example $t = 1$ to $m$:

    1. Use forward propagation to initialize activation units of all layers $(a^{(l)} \hspace{4mm} for \hspace{4mm} l=2,3,…,L)$.
    2. Use backpropagation to get the errors for all layers $(\delta^{(l)} \hspace{4mm} for \hspace{4mm} l=L,L-1,......,2)$.
    3. Update the accumulator matrices for each layer: $ \Delta^{(l)}\_{i,j} := \Delta^{(l)}\_{i,j} + a^{(l)}\_j \delta^{(l+1)}\_i$ or in the vectorized form: $ \Delta^{(l)} = \Delta^{(l)} + \delta^{(l+1)} (a^{(l)})^T$.
    
    B. Once all the training set samples have been iterated through, update the main derivative as 
    
    $$
    D^{(l)}\_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}\_{i,j} + \lambda\Theta^{(l)}\_{i,j}\right) \hspace{4mm}if\hspace{4mm} j \neq 0
    $$
    
    $$
    D^{(l)}\_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}\_{i,j}\right) \hspace{4mm} if \hspace{4mm} j = 0
    $$
    
    The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $\frac \partial {\partial \Theta\_{ij}^{(l)}} J(\Theta) = D^{(l)}\_{i,j}$.
    
### 1c: Backpropagation: Intuition

* Shows how back-propagation is the reverse of forward propagation. In forward propagation, the activation units of each layer (i.e., $a^{(l)}$) are propagated to the next layer. In backpropagation, the error units of each layer (i.e., $\delta^{(l)}$) are propagated to the previous layer.

## Lecture 2: Backpropagation in practice

### 2a: Implementation note: Unrolling parameters

* With neural networks, we are working with sets of matrices: $\Theta^{(l)}$ and $D^{(l)}$. In order to use optimizing functions such as "fminunc()" in Matlab, we will want to "unroll" all the elements and put them into one long vector: thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]. 

* To recover the original matrices from the unrolled matrices (to use in the vectorized implementations), we can use reshape.

### 2b: Gradient checking

* Gradient checking can be used to verify that back-propagation is working correctly to calculate the partial derivatives of the cost function w.r.t the model parameters.

* We can compute

$$\dfrac{\partial}{\partial\Theta\_j}J(\Theta) \approx \dfrac{J(\Theta\_1, \dots, \Theta\_j + \epsilon, \dots, \Theta\_n) - J(\Theta\_1, \dots, \Theta\_j - \epsilon, \dots, \Theta\_n)}{2\epsilon} ....... {\epsilon = 10^{-4}}$$

* This when computed for $j = 1..n$ gives us the grad_approx vector. Once we compute our gradApprox vector, we can check that gradApprox ≈ deltaVector.


### 2c: Random Initialization

* Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly.

* Hence, we initialize each $\Theta^{(l)}\_{ij}$ to a random value between $[-\epsilon,\epsilon]$. This is called symmetry breaking.

### 2d: Putting it all together

* First, pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.

a) Number of input units = dimension of features $x^{(i)}$  
b) Number of output units = number of classes  
c) Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)  
d) Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.  

* Training the neural network:

a) Randomly initialize the weights.  
b) Implement forward propagation to get the activation units of each layer and the output $h\_\Theta (x^{(i)})$.  
c) Implement the cost function.  
d) Implement backpropagation to compute partial derivatives.  
e) Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.  
f) Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.  

* However, keep in mind that $J(\Theta)$ is theoretically not convex and thus we can end up in a local minimum instead. This is usually not a concern in practice.