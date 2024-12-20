---
title: "Matrix Inverse, Deriving the Gradient for the Backward Pass"
description: >-
  Obtaining the gradient of the matrix inverse
#author: Josh Levy-Kramer
date: 2024-11-22 12:01:00 +0000
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

Here, I will derive the gradients of a matrix inverse used for backpropagation in deep learning models. I will use tensor calculus and index notation - see my article [The Tensor Calculus You Need for Deep Learning]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}) for more information.

Given $Y=X^{-1}$, we know from the definition of an inverse matrix that $I=X X^{-1}=X Y$ (assuming $X$ is square and the inverse of $X$ exists). We convert this to index notation (the indices $i$, $j$ and $k$ must be the same size):

$$
\delta_{i j}=x_{i k} y_{k j}
$$

First, we find the derivative with respect to $X$ and use the product rule:

$$
\begin{align*}
\frac{\partial \delta_{ij}}{\partial x_{pq}} &= \frac{\partial x_{ik}y_{kj}}{\partial x_{pq}} &\;\;\textrm{(no sum i, j, p, q)} \\ 
0 &= \frac{\partial x_{ik}}{\partial x_{pq}}y_{kj} + x_{ik}\frac{\partial y_{kj}}{\partial x_{pq}} \\ 
0 &= \delta_{ip}\delta_{kq}y_{kj} + x_{ik}\frac{\partial y_{kj}}{\partial x_{pq}} \\ 
0 &= \delta_{ip}y_{qj} + x_{ik}\frac{\partial y_{kj}}{\partial x_{pq}} \\ 
x_{ik}\frac{\partial y_{kj}}{\partial x_{pq}} &= -\delta_{ip}y_{qj}
\end{align*}
$$

We then multiply by the inverse matrix $y_{n i}$, we must use a new free index $n$ for the first axis and contract on the second axis $i$:

$$
\begin{aligned}
y_{n i} x_{i k} \frac{\partial y_{k j}}{\partial x_{p q}} & =-y_{n i} \delta_{i p} y_{q j} \\
\delta_{n k} \frac{\partial y_{k j}}{\partial x_{p q}} & =-y_{n p} y_{q j} \\
\frac{\partial y_{n j}}{\partial x_{p q}} & =-y_{n p} y_{q j}
\end{aligned}
$$

So the gradient of the inverse matrix with respect to itself is an order-4 tensor whereby every combination of elements of the inverse are multiplied together. This is similar to a [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) but we don't "flatten" the result into 2 dimensions.

Next, to obtain the gradient for backpropagation we assume $y_{i j}$ is an input of a scalar function $l$ and we are provided with the gradients of $l$ with respect to $y_{i j}$. Then to derive the gradients for backpropagation, we apply the chain rule:


$$
\begin{aligned}
\frac{\partial l}{\partial x_{p q}} & =\frac{\partial l}{\partial y_{n j}} \frac{\partial y_{n j}}{\partial x_{p q}} \\
& =\frac{\partial l}{\partial y_{n j}}\left(-y_{n p} y_{q j}\right) \\
& =-y_{n p} \frac{\partial l}{\partial y_{n j}} y_{q j}
\end{aligned}
$$



And we can convert it back to matrix notation:


$$
\begin{aligned}
{\left[\frac{\partial l}{\partial X}\right]_{p q} } & =-\left[Y^{T}\right]_{p n}\left[\frac{\partial l}{\partial Y}\right]_{n j}\left[Y^{T}\right]_{j q} \\
\frac{\partial l}{\partial X} & =-\left(X^{-1}\right)^{T} \frac{\partial l}{\partial Y}\left(X^{-1}\right)^{T}
\end{aligned}
$$



## PyTorch Implementation

We can check the result by comparing the equation above with PyTorch's built-in autograd output:

```python
import torch

# Create random input
torch.manual_seed(42)
K = 256
x = torch.rand((K, K), dtype=torch.float32, requires_grad=True)
dldy = torch.rand((K, K), dtype=torch.float32)

# Run forward and backward pass using built-in function
y = x.inverse()
y.backward(dldy)

# Calculate gradients using above equations
# Note: I use brackets to specify the order of the matmuls
# to be consistent with how PyTorch calculate it
dldx = -y.T @ (dldy @ y.T)

# Compare with PyTorch
torch.testing.assert_close(dldx, x.grad)
```

## Next

Further examples of calculating gradients using tensor calculus and index notation can be found on the [intro page]({% link _tabs/gradients for backpropagation.md %}).
