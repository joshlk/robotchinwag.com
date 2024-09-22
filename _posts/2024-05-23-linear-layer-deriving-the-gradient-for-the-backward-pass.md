---
title: "Linear Layer, Deriving the Gradient for the Backward Pass"
description: >-
  Deriving the gradient for the backward pass for the linear layer using tensor calculus
#author: Josh Levy-Kramer
date: 2024-05-23 12:01:00 +0000
categories:
  - AI
  - Gradients for Backpropagation
tags:
  - ai
  - deep learning
  - maths
  - backpropagation
  - tensor calculus
  - index notation
pin: false
math: true
---

The [linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html), a.k.a. [dense layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) or [fully-connected layer](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html), is everywhere in deep learning and forms the foundation to most neural networks. [PyTorch defines the linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) as:

$$
Y = XA^T + \hat{b}
$$

Whereby the tensors and their shapes are:

* Input $X$: (∗, in_features) 
* Weights $A$: (out_features,in_features)
* Bias $\hat{b}$: (out_features)
* Output $Y$: (∗, out_features)

The "∗" means any number of dimensions. PyTorch is a bit unusual in that it takes the transpose of the weights matrix $A$ before multiplying it with the input $X$. The hat over $\hat{b}$ indicates it's a vector.

In this article, we will derive the gradients used for backpropagation for the linear layer, function used when calling [`Y.backwards()`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html) on the output tensor `Y`. We will use [index notation]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#index-notation) to express the function more precisely and [tensor calculus]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#tensor-calculus) to calculate the gradients.

To derive the gradient *for each input*, we proceed as follows:

1. Translate the function into index notation.
2. Calculate the derivative with respect to the output.
3. Use the chain rule to determine the gradients of $l$ with respect to each of the inputs of the function. To do this we first must assume the output is a dependency of a downstream scalar function $l$ and we are provided with the gradients of $l$ with respect to the output.

## Using Index Notation

To keep it simple, we are going to assume there is only one "∗" dimension, and it's easy enough to extend the reasoning to more. We can express the linear layer using [index notation]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#index-notation) like so:

$$
y_{ij} = x_{ik}(a_{jk})^T + \mathbf{1}_ib_j
$$

In the first expression, the $k$ index is repeated, meaning it's a dummy index, and the two matrices $x_{ik}$ and $(a_{jk})^T$ are multiplied together, contracting the $k$ index. In the second expression, the one-tensor $ \mathbf{1}_i$ is used to broadcast the vector $b_j$ into a matrix to be added to the first expression.

## Gradients of all the inputs

Next, we want to calculate the gradient of the input tensors $x_{ik}$, $a_{jk}$ and $b_j$.

### The gradient of $X$

Let's first obtain the derivative with respect to $x_{ik}$. We need to remember to use new free indices for the derivative operator:

$$
\frac{\partial y_{i j}}{\partial x_{p q}} = \frac{\partial}{\partial x_{p q}} \left (x_{ik}(a_{jk})^T + \mathbf{1}_ib_j \right)
$$

The second term with the bias $b_j$ is independent of $x_{p q}$ and so that's zero. In the first term $a_{jk}$ is just a factor and so can be moved outside the operator:

$$
\frac{\partial y_{i j}}{\partial x_{p q}} = \frac{\partial x_{i k} }{\partial x_{p q}} (a_{j k})^T
$$

From the [rules of tensor calculus]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#tensor-derivative-operator), we know that the derivative of a variable with itself equals a product of Krockener deltas:

$$
\frac{\partial y_{i j}}{\partial x_{p q}} = \delta_{i p} \delta_{k q} (a_{j k})^T
$$

We can then contract $k$ index to obtain:

$$
\frac{\partial y_{i j}}{\partial x_{p q}} = \delta_{i p} (a_{j q})^T
$$

This is an order-4 tensor, i.e. a tensor with 4 dimensions, and so can't be expressed using matrix notation. However, the tensor is only non-zero when $i == p$ due to the Krockener delta. Fortunately, the gradient used for backpropagation is a lower-order tensor, and we can use matrix notation. To do that lets first assume $y_{ij}$ is an input of a scalar function $l$ and we are provided with the gradients of $l$ with respect to $y_{ij}$. Then to derive the gradients for backpropagation, we apply the chain rule:

$$
\begin{aligned}
\frac{\partial l}{\partial x_{p q}} & =\frac{\partial l}{\partial y_{i j}} \frac{\partial y_{i j}}{\partial x_{p q}} \\
& =\frac{\partial l}{\partial y_{i j}} \delta_{i p} (a_{j q})^T \\
& =\frac{\partial l}{\partial y_{p j}} (a_{j q})^T
\end{aligned}
$$

We can convert it to matrix notation like so (the square brackets indicate taking the elements of the matrix):

$$
\begin{aligned}
{\left[\frac{\partial l}{\partial X}\right]_{p q} } & =\left[\frac{\partial l}{\partial Y}\right]_{p j}\left[A\right]_{j q} \\
\therefore \frac{\partial l}{\partial X} & =\frac{\partial l}{\partial Y} A
\end{aligned}
$$

Therefore the gradient is calculated by multiplying the gradient of the loss $l$ with respect to the output $Y$ with the weights $A$.

### Gradient of $A$

First, obtain the derivative with respect to $A$:

$$
\begin{aligned}
\frac{\partial y_{i j}}{\partial a_{p q}} & =\frac{\partial x_{i k} (a_{j k})^T}{\partial a_{p q}} \\
& =x_{i k} \frac{\partial a_{k j}}{\partial a_{p q}} \\
& =x_{i k} \delta_{k p} \delta_{j q} \\
& =x_{i p} \delta_{j q}
\end{aligned}
$$

Notice how we use the fact that $(a_{j k})^T = a_{k j}$. Then, we obtain the backpropagated gradient by assuming a downstream loss $l$ consumes the output $y$:

$$
\begin{aligned}
\frac{\partial l}{\partial a_{p q}} & =\frac{\partial l}{\partial y_{i j}} \frac{\partial y_{i j}}{\partial a_{p q}} \\
& =\frac{\partial l}{\partial y_{i j}} x_{i p} \delta_{j q} \\
& =\frac{\partial l}{\partial y_{i q}} x_{i p}
\end{aligned}
$$

We can then convert this back to matrix notation:

$$
\begin{aligned}
{\left[\frac{\partial l}{\partial A}\right]_{p q} } & =\left[X^{T}\right]_{p i}\left[\frac{\partial l}{\partial Y}\right]_{i q} \\
\therefore \frac{\partial l}{\partial A} & =X^{T} \frac{\partial l}{\partial Y}
\end{aligned}
$$

Therefore, the gradient is calculated by multiplying the transpose of the input $X$ with the gradient of the loss $l$ with respect to the output $Y$.

### Gradient of $\hat{b}$

First we calculate the gradient of the output with respect to the bias:

$$
\begin{aligned}
\frac{\partial y_{i j}}{\partial b_{p}} & = \frac{\partial}{\partial b_{p}} \left (x_{ik}(a_{jk})^T + \mathbf{1}_ib_j \right) \\
&= \frac{\partial \left( \mathbf{1}_i b_j \right)}{\partial b_{p}}  \\
&= \mathbf{1}_i \frac{\partial b_j}{\partial b_{p}}
\end{aligned}
$$

Again, using the [rules of tensor calculus]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#tensor-derivative-operator), we know that the derivative of a variable with itself equals a product of Krockener deltas:

$$
\frac{\partial y_{i j}}{\partial b_{p}} = \mathbf{1}_i \delta_{jp}
$$
Finally, we derive the gradient used for backpropagation by assuming a downstream loss $l$ consumes the output $y$:

$$
\begin{aligned}
\frac{\partial l}{\partial b_p} & =\frac{\partial l}{\partial y_{i j}} \frac{\partial y_{i j}}{\partial b_{p}} \\
& =\frac{\partial l}{\partial y_{i j}} \mathbf{1}_i \delta_{jp} \\
& =\frac{\partial l}{\partial y_{i p}} \mathbf{1}_i
\end{aligned}
$$

Simply put, this means we sum the gradient in the $i$ dimension to obtain the backpropagated gradient of the bias:

$$
\frac{\partial l}{\partial b_p} = \sum_i \frac{\partial l}{\partial y_{i p}}
$$

Or in matrix notation, you can express this using a vector filled with ones:

$$
\frac{\partial l}{\partial \hat{b}} = \begin{pmatrix}
1 & \cdots & 1
\end{pmatrix} \frac{\partial l}{\partial Y}
$$

## Next

If you would like to read more about calculating gradients using tensor calculus and index notation, please have a look at the [series introduction]({% link _tabs/gradients for backpropagation.md %}) or [The Tensor Calculus You Need for Deep Learning]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}).
