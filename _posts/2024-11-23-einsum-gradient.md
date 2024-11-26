---
title: "Einsum, Deriving the Gradient for the Backward Pass"
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

[Einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html) allows you to sum and broadcast across arbitrary dimensions for a set of tensors. It lets you neatly describe dot products, outer products, transposes and various types of multiplications. Some even proclaim [Einsum is all you need!](https://rockt.github.io/2018/04/30/einsum) (Although there are lots of things you can't do with einsum). With Einsum you describe the operation using a syntax which is similar to [index notation]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#index-notation).

Here we are going to explore how to derive the backpropagated gradient for the einsum function, whereby we contract (sum) an arbitrary set of dimensions. I will use tensor calculus and index notation - see my article [The Tensor Calculus You Need for Deep Learning]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}) for more information.

We are provided with two input tensors $x$ and $y$ and we know:

* $x$ has a set of dimensions that we will broadcast using einsum, which we will represent with free indices $a \dots b$.
* $y$ has a set of dimensions that we will broadcast using einsum, which we will represent with free indices $c \dots d$.
* A set of dimensions will be contracted (summed) using einsum and represented using dummy indices $p \dots q$.

The output tensor $z$ is computed like so:


$$
z_{a \dots b c \dots d} = x_{a \dots b p \dots q}y_{c \dots d p \dots q }
$$

In the above equation, we have separated the free and dummy indices to keep the notation cleaner but they can be mixed together. If we ignore the ellipsis, the above equation is equivalent to the einsum expression `abpq,cdpq->abcd`.

First, let's consider the derivative of $z$ with respect to the input $x$. We need to introduce new free indices for $x$ and I represent this using the prime symbol:


$$
\begin{aligned}
\frac{\partial z_{a \dots b c \dots d}}{\partial x_{a' \dots b' p' \dots q'}}
&= \frac{\partial x_{a \dots b p \dots q}}{\partial x_{a' \dots b' p' \dots q'}} y_{c \dots d p \dots q } \\
&= \delta_{aa'}\dots\delta_{bb'}\delta_{pp'}\dots\delta_{qq'} y_{c \dots d p \dots q }
\end{aligned}
$$

Next, we want to obtain the backpropagated gradient. To obtain this, we assume a downstream scalar function $l$ consumes $z$ and we are provided with the gradient of $l$ with respect to $z$. We then use the chain rule to obtain the gradient of $l$ with respect to $x$:


$$
\begin{aligned}
\frac{\partial l}{\partial x_{a' \dots b' p' \dots q'}}
&= \frac{\partial l}{\partial z_{a \dots b c \dots d}}\frac{\partial z_{a \dots b c \dots d}}{\partial x_{a' \dots b' p' \dots q'}}\\
&= \frac{\partial l}{\partial z_{a \dots b c \dots d}}\delta_{aa'}\dots\delta_{bb'}\delta_{pp'}\dots\delta_{qq'} y_{c \dots d p \dots q } \\

&= \frac{\partial l}{\partial z_{a' \dots b' c \dots d}} y_{c \dots d p' \dots q' }

\end{aligned}
$$

So this tells us that to obtain the gradient we contract the common indices between the gradient and $y$ and broadcast the others.

## Example

To provide a concrete example, say we have two tensors $x_{ijk}$ and $y_{ip}$. If we use the einsum expression `ijk,ip->jkp` to obtain the output $z$ , this is equivalent to:


$$
z_{jkp} = x_{ijk}y_{ip}
$$


To obtain the gradient we use the equation above:




$$
\frac{\partial l}{\partial x_{i'j'k'}}
= \frac{\partial l}{\partial z_{j'k'p}} y_{i'p}
$$


Which is equivalent to the einsum expression: `jkp,ip->ijk` 

We can test this using PyTorch:

```python
import torch

# Create random inputs
torch.manual_seed(42)
I, J, K, P = 16, 32, 64, 128
x = torch.rand((I, J, K), dtype=torch.float32, requires_grad=True)
y = torch.rand((I, P), dtype=torch.float32, requires_grad=True)
dldz = torch.rand((J, K, P))

# Run forward and backward pass using built-in function
z = torch.einsum('ijk,ip->jkp', x, y)
z.backward(dldz)

# Obtain gradient using above equation
dldx = torch.einsum('jkp,ip->ijk', dldz, y)

# Compare with PyTorch
torch.testing.assert_close(dldx, x.grad)
```

## General PyTorch Implementation

We can generalise the result. First lets generate two random input tensors and a random einsum expression:

```python
import torch
import string
import random

torch.manual_seed(42)
random.seed(42)

# Number of free and dummy indices for each tensor
x_n_free = 2
y_n_free = 3
n_dummy = 4

# Assign a letter to each index
alphabet = list(string.ascii_lowercase)
x_free_idx = alphabet[:x_n_free]
y_free_idx = alphabet[x_n_free:x_n_free+y_n_free]
dummy_idx = alphabet[x_n_free+y_n_free:x_n_free+y_n_free+n_dummy]

# Shuffle the order of the dimensions
x_idx_order = x_free_idx + dummy_idx
y_idx_order = y_free_idx + dummy_idx
z_idx_order = x_free_idx + y_free_idx
random.shuffle(x_idx_order)
random.shuffle(y_idx_order)
random.shuffle(z_idx_order)

# Assign a random size to each dimension
sizes = {idx: random.randint(2, 32) for idx in x_free_idx+y_free_idx+dummy_idx}

# Generate random values
x = torch.rand(size=[sizes[idx] for idx in x_idx_order], requires_grad=True, dtype=torch.float32)
y = torch.rand(size=[sizes[idx] for idx in y_idx_order], requires_grad=True, dtype=torch.float32)
dldz = torch.rand(size=[sizes[idx] for idx in z_idx_order], dtype=torch.float32)

# Create einsum expression
z_eignsum_equation = ''.join(x_idx_order) + ',' + ''.join(y_idx_order) + '->' + ''.join(z_idx_order)
```

Now let's run einsum to obtain the output, use the equations above to calculate the gradients and compare the results with PyTorch.

```python
# Run einsum to obtain output z
z = torch.einsum(z_eignsum_equation, x, y)

# Obtain the gradients using autograd
z.backward(dldz)

# Obtain the gradients using the general equations from above
x_grad_eignsum_equation = ''.join(z_idx_order) + ',' + ''.join(y_idx_order) + '->' + ''.join(x_idx_order)
dldx = torch.einsum(x_grad_eignsum_equation, dldz, y)

y_grad_eignsum_equation = ''.join(z_idx_order) + ',' + ''.join(x_idx_order) + '->' + ''.join(y_idx_order)
dldy = torch.einsum(y_grad_eignsum_equation, dldz, x)

# Test to see if the results are the same
torch.testing.assert_close(dldx, x.grad)
torch.testing.assert_close(dldy, y.grad)
```

## Next

Further examples of calculating gradients using tensor calculus and index notation can be found on the [intro page]({% link _tabs/gradients for backpropagation.md %}).
