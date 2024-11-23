---
title: "Cross-entropy loss (softmax and negative log-likelihood loss) gradient for deep learning"
description: >-
  Obtaining the gradient of the Cross-entropy loss (softmax and negative log-likelihood loss function
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



[Cross-entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) is a common loss used for classification tasks in deep learning - including transformers. It is defined as the [softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) function followed by the [negative log-likelihood loss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html). Here, I will walk through how to derive the gradient of the cross-entropy loss used for the backward pass when training a model. I will use tensor calculus and index notation - see my article [The Tensor Calculus You Need for Deep Learning]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}) for more information.

Say we have an input vector of logits $\hat{x}$ and vector of target classes $\hat{s}$, we can define the cross-entropy loss using index notation:
$$
\begin{aligned}
s_{i} & =\frac{e^{x_{i}}}{\mathbf{1}_{j} e^{x_{j}}} \\
c & =-\log s_{T}
\end{aligned}
$$

The first equation calculates the softmax: for every value in $\hat{x}$ we divide the exponent by the sum of the exponent of all values. The second equation calculates the negative log-likelihood loss of the softmax: $T$ is the index corresponding to the correct target label. Note that $T$ is a constant and not a free or dummy index.

First, we need to derive the Jacobian tensor of the function. Let's start with the denominator in the softmax:

$$
\begin{aligned}
\sigma & =\mathbf{1}_{j} e^{x_{j}} \\
\frac{\partial \sigma}{\partial x_{n}} & =\mathbf{1}_{j} \frac{\partial e^{x_{j}}}{\partial x_{n}} \\
& =\mathbf{1}_{j} \frac{\partial e^{x_{j}}}{\partial x_{j}} \frac{\partial x_{j}}{\partial x_{n}} \\
& =\mathbf{1}_{j} e^{x_{j}} \delta_{j n} \\
& =e^{x_{n}}
\end{aligned}
$$

Notice that we can drop $$ \mathbf{1}_{j} $$ as $j$ is a free index, and $$ \mathbf{1}_{j} $$ always equals 1.

To differentiate the softmax tensor with respect to $x$, we use the quotient rule and simplify the expression by reusing the definition of the softmax:

$$
\begin{aligned}
\frac{\partial s_{i}}{\partial x_{n}} & =\frac{1}{\sigma^{2}}\left(\frac{\partial e^{x_{i}}}{\partial x_{n}} \sigma-e^{x_{i}} \frac{\partial \sigma}{\partial x_{n}}\right) \\
& =\frac{1}{\sigma^{2}}\left(e^{x_{i}} \delta_{i n} \sigma-e^{x_{i}} e^{x_{n}}\right) \\
& =s_{i} \delta_{i n}-s_{i} s_{n}
\end{aligned}
$$

And we move on to the negative log-likelihood loss:

$$
\begin{aligned}
\frac{\partial c}{\partial s_{i}} & =-\frac{\partial \log s_{T}}{\partial s_{i}} \\
& =-\frac{1}{s_{T}} \delta_{i T}
\end{aligned}
$$

Note that as $T$ is a constant, not a dummy index, the expression is $-1 / s_{T}$ when $i=T$ and zero otherwise (it is not a summation).

Putting the two expressions together to get the complete gradient:

$$
\begin{aligned}
\frac{\partial c}{\partial x_{n}} & =\frac{\partial c}{\partial s_{i}} \frac{\partial s_{i}}{\partial x_{n}} \\
& =\left(-\frac{1}{s_{T}} \delta_{i T}\right)\left(s_{i} \delta_{i n}-s_{i} s_{n}\right) \\
& =-\frac{1}{s_{T}}\left(s_{T} \delta_{T n}-s_{T} s_{n}\right) \\
& =s_{n}-\delta_{T n}
\end{aligned}
$$

It's interesting to note that because of the influence of normalising all values by $\sigma$, all logits have a non-zero gradient even if they do not correspond to the true label.

Then, deriving the backpropagated gradient is trivial:

$$
\begin{aligned}
\frac{\partial l}{\partial x_{n}} & =\frac{\partial l}{\partial c} \frac{\partial c}{\partial x_{n}} \\
& =\frac{\partial l}{\partial c}\left(s_{n}-\delta_{T_{n}}\right)
\end{aligned}
$$

It might be the case that we start backpropagation using the cross-entropy loss, in that case $l=c$ and $\partial l / \partial c=1$.

## PyTorch Implementation

We can check the result by comparing the equation above with PyTorch's built-in autograd output. I have generalised the above equation for a batch of results of size (N):

```python
import torch
from torch import nn
import torch.nn.functional as F

# Create random input
torch.manual_seed(42)
N = 128
num_classes = 256
x = torch.rand((N, num_classes), dtype=torch.float32, requires_grad=True)
target = torch.randint(high=num_classes-1,  size=(N,))
dldc = torch.rand((N,), dtype=torch.float32)

# Run forward and backward pass using built-in function
loss_layer = nn.CrossEntropyLoss(reduction='none')
c = loss_layer(x, target)
c.backward(dldc)

# Calculate gradient using above equation
s = torch.softmax(x, dim=1)
dldx = dldc.unsqueeze(1)*(s - F.one_hot(target, num_classes=num_classes))

# Compare with PyTorch
torch.testing.assert_close(dldx, x.grad)
```

## Next

Further examples of calculating gradients using tensor calculus and index notation can be found on the [intro page]({% link _tabs/gradients for backpropagation.md %}).
