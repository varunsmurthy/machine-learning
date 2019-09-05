# Week 4: Neural Networks (Representation)

## Lecture 1: Motivations

### 1a: Non-Linear Hypotheses

* Regression and classification problems can sometimes have a large number of features (e.g., consider a 100 x 100 pixel image classification problem, where each pixel is a feature)and often need quadratic terms as features (i.e. $x_1^2, x_1x_2,..$. However, the number of term scales as $n^2/2$ and becomes very large for large $n$ (where $n$ is the number of features). 

### 1b: Neurons and the brain

* The cortex in the brain can reprogram itself to do differemt things (i.e., the part of the brain that is thought to understand sounds can reprogram itself to understand vision if the input is changed).

## Lecture 2: Neural Networks

### 2a: Model representation

* A neuron in the brain is made up of dendrites (inputs), axon (output), and a body (computation).

* An artificial neuron is made up of input wires, output wires, and an activation unit. An activation unit (a) or a neuron is characterized by a logistic or sigmoid activation function (similar to logistic regression). In this situation, our "theta" parameters are sometimes called "weights". Also, in this model our $x_0$ input node is sometimes called the "bias unit." It is always equal to 1.

* An artificial neural network is made up of multiple layers of neurons connected to each other. The first layer (layer 1) is called the input layer, the last layer is called the output layer, and the layers in between are called hidden layers.

* Terminology: 
$$
\begin{align*}& a_i^{(j)} = \text{"activation" of unit $i$ in layer $j$} \newline& \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{align*}$$

* If we had one hidden layer, it would look like

$$
\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \newline a_3^{(2)} \newline \end{bmatrix}\rightarrow h_\theta(x)
$$

* The values of the activation units or neurons would be

$$
\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
$$

* $ \text{If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j + 1)$.} $

### 2b: Model representation II

* Vectorized mplementation: First, we assign the input layer (which is a $n+1 \times 1$ vector i.e., $x_0, x_1, x_2, ..x_n$) as the first hidden layer (i.e., $a^{(1)}$). Then the vectorized implementation for the subsequent layers is 

$$
a^{(j)} = g(\Theta^{(j-1)}a^{(j-1)})
$$

where $g$ is the logistic or sigmoid function. 

* We can then add a bias unit (equal to 1) to layer j after we have computed $a^{(j)}$. This will be element $a_0^{(j)}$ and will be equal to 1 when computing the subsequent layer.

* Notice that in the final step, when describing the activation unit of the output layer, we are doing exactly the same thing as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

## Lecture 3: Applications

### 3a: Examples and Intuitions

* Representing basic logical gates (AND, OR, NOT) using neural netwroks:

* The AND gate can be represented as

$$
\begin{align*}\begin{bmatrix}x_0 \newline x_1 \newline x_2\end{bmatrix} \rightarrow\begin{bmatrix}g(z^{(2)})\end{bmatrix} \rightarrow h_\Theta(x)\end{align*}
$$

where

$$
\Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20\end{bmatrix}
$$

### 3b: Examples and Intuitions II

* Some logical expressions (e.g., XNOR) cannot be easily represented using single layer neaural networks. In these cases, multiple layers can be used (e.g., NOR = AND OR NOR). So in this example, $a_1^{(2)}$ and $a_1^{(2)}$ are AND and NOR, respectively, and $a_1^{(3)}$ is OR.

* Short video showing how multiple layers of neural networks can be used for handwriting recognition.

### 3c: Multi-class classification

* The output layer will now have k units, where k is the number of classes. For a classification problem with 4 classes, if the input object is of class type 3, the output of the neural network will look like

$$
h_\Theta(x) =\begin{bmatrix}0 \newline 0 \newline 1 \newline 0 \newline\end{bmatrix}
$$