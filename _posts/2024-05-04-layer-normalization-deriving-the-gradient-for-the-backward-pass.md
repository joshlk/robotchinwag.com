---
title: "Layer Normalization, Deriving the Gradient for the Backward Pass"
description: >-
  Obtaining the gradient of the layer normalization layer
#author: Josh Levy-Kramer
date: 2024-05-04 12:01:00 +0000
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

This post explains how to calculate the gradients of layer normalisation used for backpropagation using [tensor calculus]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#tensor-calculus) and [index notation]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#index-notation). It is part of a [series]({% link _tabs/gradients for backpropagation.md %}) on differentiating and calculating gradients in deep learning. This example is quite long and involved but combines the different concepts presented in the article series. If you have not done so, be sure to become familiar with the [previous examples]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#example-element-wise-functions) first.

PyTorch defines the layer normalization operation for an input matrix $X$, with shape batch size $(B)$ by hidden size $(H)$, as:

$$
y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta
$$

Where the mean $\mathrm{E}[x]$ and variance $\operatorname{Var}[x]$ are calculated for each sample in a batch, and $\gamma$ and $\beta$ are learnable vector weights with lengths equal to the hidden size. $\epsilon$ is a constant usually equal to $1 \mathrm{e}-05$.

[As shown previously]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#example-layer-normalisation-using-index-notation), we can represent this using index notation:
$$
\begin{aligned}
m_{b} & =\frac{1}{H} \mathbf{1}_{h} x_{b h} \\
v_{b} & =\frac{1}{H} \mathbf{1}_{h}\left(x_{b h}-\mathbf{1}_{h} m_{b}\right)^{2} \\
y_{b h} & =\frac{x_{b h}-\mathbf{1}_{h} m_{b}}{\sqrt{v_{b}+\epsilon}} \gamma_{h}+\mathbf{1}_{b} \beta_{h}
\end{aligned}
$$

To make the problem more manageable, we are going to define additional intermediate tensor functions $\mu_{b h}$ and $\sigma_{b}$:

$$
\begin{aligned}
m_{b} & =\frac{1}{H} \mathbf{1}_{h} x_{b h} \\
\mu_{b h} & =x_{b h}-\mathbf{1}_{h} m_{b} \\
v_{b} & =\frac{1}{H} \mathbf{1}_{h} \mu_{b h}^{2} \\
\sigma_{b} & =\sqrt{v_{b}+\epsilon} \\
y_{b h} & =\frac{\mu_{b h}}{\sigma_{b}} \gamma_{h}+\mathbf{1}_{b} \beta_{h}
\end{aligned}
$$

The tensor functions above have the following dependency graph:

![dependency graph](/assets/img/layer_norm.svg){: width="250" }

## Gradient of Weights

Let's start with the easier gradients $\gamma$ and $\beta$:

$$
\begin{aligned}
& \frac{\partial y_{b h}}{\partial \gamma_{q}}=\frac{\mu_{b h}}{\sigma_{b}} \delta_{h q} \\
& \frac{\partial y_{b h}}{\partial \beta_{q}}=\mathbf{1}_{b} \delta_{h q}
\end{aligned}
$$

Secondly, we find the backpropagated gradient:

$$
\begin{aligned}
\frac{\partial l}{\partial \gamma_{q}} & =\frac{\partial l}{\partial y_{b h}} \frac{\partial y_{b h}}{\partial \gamma_{q}}=\frac{\partial l}{\partial y_{b q}} \frac{\mu_{b q}}{\sigma_{b}} \\
\frac{\partial l}{\partial \beta_{q}} & =\frac{\partial l}{\partial y_{b h}} \frac{\partial y_{b h}}{\partial \beta_{q}}=\frac{\partial l}{\partial y_{b q}} \mathbf{1}_{b}
\end{aligned}
$$

## Gradient of Input $X$

Directly calculating the derivative of $y_{b h}$ with respect to $x_{p q}$ is quite complex and is an order-4 tensor. However, we don't need to construct this tensor fully since we can backpropagate the loss after each intermediate tensor function, simplifying the process. The backpropagated gradient is simpler because the loss is a scalar, meaning the gradient is, at most, an order-2 tensor.

To accomplish this, we'll start at the end of the dependency graph and calculate the Jacobian tensor at each intermediate stage, followed by calculating the backpropagated gradient. The goal is to obtain an expression of $\partial l / \partial x_{p q}$ in terms of $\partial l / \partial y_{p q}$.

## Gradient of $\sigma$

The derivative of $y_{b h}$ with respect to $\sigma_{p}$:

$$
\begin{aligned}
\frac{\partial y_{b h}}{\partial \sigma_{p}} & =\frac{\partial}{\partial \sigma_{p}}\left(\mu_{b h} \sigma_{b}^{-1} \gamma_{h}+\mathbf{1}_{b} \beta_{h}\right) \\
& =-\mu_{b h} \sigma_{b}^{-2} \gamma_{h} \delta_{b p}
\end{aligned}
$$

And the backpropagated gradient:

$$
\begin{aligned}
\frac{\partial l}{\partial \sigma_{p}} & =\frac{\partial l}{\partial y_{b h}} \frac{\partial y_{b h}}{\partial \sigma_{p}} \\
& =\frac{\partial l}{\partial y_{b h}}\left(-\mu_{b h} \sigma_{b}^{-2} \gamma_{h} \delta_{b p}\right) \\
& =-\frac{\partial l}{\partial y_{p h}} \mu_{p h} \sigma_{p}^{-2} \gamma_{h}
\end{aligned}
$$

## Gradient of $v$

The derivative of $\sigma_{b}$ with respect to $v_{p}$:

$$
\begin{aligned}
\frac{\partial \sigma_{b}}{\partial v_{p}} & =\frac{\partial}{\partial v_{p}}\left[\left(v_{b}+\epsilon\right)^{0.5}\right] \\
& =\frac{1}{2}\left(v_{b}+\epsilon\right)^{-0.5} \delta_{b p} \\
& =\frac{\delta_{b p}}{2 \sigma_{b}}
\end{aligned}
$$

The backpropagated gradient:

$$
\begin{aligned}
\frac{\partial l}{\partial v_{p}} & =\frac{\partial l}{\partial \sigma_{b}} \frac{\partial \sigma_{b}}{\partial v_{p}} \\
& =\frac{\partial l}{\partial \sigma_{b}} \frac{\delta_{b p}}{2 \sigma_{b}} \\
& =\frac{\partial l}{\partial \sigma_{p}} \frac{1}{2 \sigma_{p}}
\end{aligned}
$$

Substituting in $\partial l / \partial \sigma_{p}$ from the previous step:

$$
\begin{aligned}
\frac{\partial l}{\partial v_{p}} & =\frac{\partial l}{\partial \sigma_{p}} \frac{1}{2 \sigma_{p}} \\
& =\left(-\frac{\partial l}{\partial y_{p h}} \mu_{p h} \sigma_{p}^{-2} \gamma_{h}\right) \frac{1}{2 \sigma_{p}} \\
& =-\frac{\partial l}{\partial y_{p h}} \frac{\mu_{p h} \gamma_{h}}{2 \sigma_{p}^{3}}
\end{aligned}
$$

## Gradient of $\mu$

The function $\mu_{b h}$ is consumed by two functions, $v_{b}$ and $y_{b h}$, therefore we need to differentiate both functions by $\mu_{b h}$. First, the derivative of $v_{b}$ with respect to $\mu_{p q}$:

$$
\begin{aligned}
\frac{\partial v_{b}}{\partial \mu_{p q}} & =\frac{\partial}{\partial \mu_{p q}}\left(\frac{1}{H} \mathbf{1}_{h} \mu_{b h}^{2}\right) \\
& =\frac{2}{H} \mathbf{1}_{h} \mu_{b h} \delta_{b p} \delta_{h q} \\
& =\frac{2}{H} \mu_{b q} \delta_{b p}
\end{aligned}
$$

Then derivative of $y_{b h}$ with respect to $\mu_{p q}$:

$$
\begin{aligned}
\frac{\partial y_{b h}}{\partial \mu_{p q}} & =\frac{\partial}{\partial \mu_{p q}}\left(\frac{\mu_{b h}}{\sigma_{b}} \gamma_{h}+\mathbf{1}_{b} \beta_{h}\right) \\
& =\frac{\gamma_{h}}{\sigma_{b}} \delta_{b p} \delta_{h q}
\end{aligned}
$$

When applying the chain rule to obtain the backpropagated gradient, we need to include contributions from both functions:

$$
\begin{aligned}
\frac{\partial l}{\partial \mu_{p q}} & =\frac{\partial l}{\partial v_{b}} \frac{\partial v_{b}}{\partial \mu_{p q}}+\frac{\partial l}{\partial y_{b h}} \frac{\partial y_{b h}}{\partial \mu_{p q}} \\
& =\frac{\partial l}{\partial v_{b}}\left(\frac{2}{H} \mu_{b q} \delta_{b p}\right)+\frac{\partial l}{\partial y_{b h}}\left(\frac{\gamma_{h}}{\sigma_{b}} \delta_{b p} \delta_{h q}\right) \\
& =\frac{\partial l}{\partial v_{p}} \frac{2 \mu_{p q}}{H}+\frac{\partial l}{\partial y_{p q}} \frac{\gamma_{q}}{\sigma_{p}}
\end{aligned}
$$

## Gradient of $m$

The derivative with respect to $m_{p}$:

$$
\begin{aligned}
\frac{\partial \mu_{b h}}{\partial m_{p}} & =\frac{\partial}{\partial m_{p}}\left(x_{b h}-\mathbf{1}_{h} m_{b}\right) \\
& =-\mathbf{1}_{h} \delta_{b p}
\end{aligned}
$$

And the backpropagated gradient:

$$
\begin{aligned}
\frac{\partial l}{\partial m_{p}} & =\frac{\partial l}{\partial \mu_{b h}} \frac{\partial \mu_{b h}}{\partial m_{p}} \\
& =\frac{\partial l}{\partial \mu_{b h}}\left(-\mathbf{1}_{h} \delta_{b p}\right) \\
& =-\frac{\partial l}{\partial \mu_{p h}} \mathbf{1}_{h}
\end{aligned}
$$

And substituting $\partial l / \partial \mu_{p h}$ derived from the previous step:

$$
\begin{aligned}
\frac{\partial l}{\partial m_{p}} & =-\mathbf{1}_{h}\left(\frac{\partial l}{\partial v_{p}} \frac{2 \mu_{p h}}{H}+\frac{\partial l}{\partial y_{p h}} \frac{\gamma_{h}}{\sigma_{p}}\right) \\
& =-\frac{\partial l}{\partial v_{p}} \frac{2}{H}\left(\mathbf{1}_{h} \mu_{p h}\right)-\frac{\partial l}{\partial y_{p h}} \frac{\gamma_{h}}{\sigma_{p}}
\end{aligned}
$$

In the first term, we have the sum $$ \mathbf{1}_{h} \mu_{p h} $$ which can be shown to equal zero:

$$
\begin{aligned}
\mathbf{1}_{h} \mu_{p h} & =\mathbf{1}_{h}\left(x_{p h}-\mathbf{1}_{h} m_{p}\right) \\
& =\mathbf{1}_{h} x_{p h}-\mathbf{1}_{h} \mathbf{1}_{h} m_{p} \\
& =H m_{p}-H m_{p} \\
& =0
\end{aligned}
$$

And so, we can simplify the above expression:

$$
\frac{\partial l}{\partial m_{p}}=-\frac{\partial l}{\partial y_{p h}} \frac{\gamma_{h}}{\sigma_{p}}
$$

## Gradient of $x$

And finally, we move onto $x_{b h}$. Two functions, $m_{b}$ and $\mu_{b h}$, consume $x_{b h}$ and so we need to consider both. First, the derivative of $m_{b}$ with respect to $x_{p q}$:

$$
\begin{aligned}
\frac{\partial m_{b}}{\partial x_{p q}} & =\frac{\partial}{\partial x_{p q}}\left(\frac{1}{H} \mathbf{1}_{h} x_{b h}\right) \\
& =\frac{\mathbf{1}_{q}}{H} \delta_{b p}
\end{aligned}
$$

And the derivative of $\mu_{b h}$ with respect to $x_{p q}$:

$$
\begin{aligned}
\frac{\partial \mu_{b h}}{\partial x_{p q}} & =\frac{\partial}{\partial x_{p q}}\left(x_{b h}-\mathbf{1}_{h} m_{b}\right) \\
& =\delta_{b p} \delta_{h q}
\end{aligned}
$$

Finally, we use the chain rule to obtain the backpropagated gradient and combine the contribution from both functions:

$$
\begin{aligned}
\frac{\partial l}{\partial x_{p q}} & =\frac{\partial l}{\partial m_{b}} \frac{\partial m_{b}}{\partial x_{p q}}+\frac{\partial l}{\partial \mu_{b h}} \frac{\partial \mu_{b h}}{\partial x_{p q}} \\
& =\left(-\frac{\partial l}{\partial y_{b h}} \frac{\gamma_{h}}{\sigma_{b}}\right)\left(\frac{\mathbf{1}_{q}}{H} \delta_{b p}\right)+\left(\frac{\partial l}{\partial v_{b}} \frac{2 \mu_{b h}}{H}+\frac{\partial l}{\partial y_{b h}} \frac{\gamma_{h}}{\sigma_{b}}\right)\left(\delta_{b p} \delta_{h q}\right) \\
& =-\frac{\mathbf{1}_{q}}{H} \frac{\partial l}{\partial y_{p h}} \frac{\gamma_{h}}{\sigma_{p}}+\frac{2}{H} \frac{\partial l}{\partial v_{p}} \mu_{p q}+\frac{\partial l}{\partial y_{p q}} \frac{\gamma_{q}}{\sigma_{p}}
\end{aligned}
$$

The goal is to obtain an expression of $\partial l / \partial x_{p q}$ in terms of $\partial l / \partial y_{p q}$, and so we substituent $\partial l / \partial v_{p}$ using the previously derived expression and rearranging the terms to obtain the final result:

$$
\begin{aligned}
\frac{\partial l}{\partial x_{p q}} & =-\frac{\mathbf{1}_{q}}{H} \frac{\partial l}{\partial y_{p h}} \frac{\gamma_{h}}{\sigma_{p}}+\frac{2}{H}\left(-\frac{\partial l}{\partial y_{p h}} \frac{\mu_{p h} \gamma_{h}}{2 \sigma_{p}^{3}}\right) \mu_{p q}+\frac{\partial l}{\partial y_{p q}} \frac{\gamma_{q}}{\sigma_{p}} \\
& =\frac{\partial l}{\partial y_{p q}} \frac{\gamma_{q}}{\sigma_{p}}-\frac{\partial l}{\partial y_{p h}} \frac{\gamma_{h}}{H}\left(\frac{\mathbf{1}_{q}}{\sigma_{p}}+\frac{\mu_{p h} \mu_{p q}}{\sigma_{p}^{3}}\right)
\end{aligned}
$$

## Conclusion

Bringing the results together:

$$
\begin{gathered}
\frac{\partial l}{\partial \gamma_{q}}=\frac{\partial l}{\partial y_{b q}} \frac{\mu_{b q}}{\sigma_{b}} \\
\frac{\partial l}{\partial \beta_{q}}=\frac{\partial l}{\partial y_{b q}} \mathbf{1}_{b} \\
\frac{\partial l}{\partial x_{p q}}=\frac{\partial l}{\partial y_{p q}} \frac{\gamma_{q}}{\sigma_{p}}-\frac{\partial l}{\partial y_{p h}} \frac{\gamma_{h}}{H}\left(\frac{\mathbf{1}_{q}}{\sigma_{p}}+\frac{\mu_{p h} \mu_{p q}}{\sigma_{p}^{3}}\right)
\end{gathered}
$$



## PyTorch Implementation

We can numerically check the above result by implementing the equations in PyTorch and numerically comparing the result to the built-in PyTorch function:

```python
import torch

# Create random inputs
torch.manual_seed(42)
B, H = 128, 256
eps = 1e-05
x = torch.rand((B, H), dtype=torch.float32, requires_grad=True)
gamma = torch.rand(H, dtype=torch.float32, requires_grad=True)
beta = torch.rand(H, dtype=torch.float32, requires_grad=True)
dldy = torch.rand((B, H), dtype=torch.float32)

# Run forward and backward pass using built-in function
y = torch.nn.functional.layer_norm(x, [H], gamma, beta)
y.backward(dldy)

# Calculate gradients using above equations
m = x.mean(axis=1)
mu = x - m.unsqueeze(1)
v = torch.mean(mu**2, axis=1)
sigma = torch.sqrt(v + eps)

dldgamma = torch.einsum('bq,bq,b->q', [dldy, mu, 1/sigma])
dldbeta = dldy.sum(axis=0)

dldx = (
    dldy*gamma.unsqueeze(0) / sigma.unsqueeze(1)
    - 1/H * torch.einsum('ph,h,p->p', [dldy, gamma, 1/sigma]).unsqueeze(1)
    - 1/H * mu * torch.einsum('ph,h,ph,p->p', [dldy, gamma, mu, sigma**(-3)]).unsqueeze(1)
)

# Compare against PyTorch
torch.testing.assert_close(dldgamma, gamma.grad)
torch.testing.assert_close(dldbeta, beta.grad)
torch.testing.assert_close(dldx, x.grad)
```

We can also implement our own custom PyTorch layer as well:

```python
class LayerNormManual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        eps = 1e-05
        assert x.dim() == 2
        B, H = x.shape
        m = x.mean(axis=1)
        mu = x - m.unsqueeze(1)
        v = torch.mean(mu**2, axis=1)
        sigma = torch.sqrt(v + eps)
        
        y = (mu/sigma.unsqueeze(1))*gamma.unsqueeze(0) + beta.unsqueeze(0)
        
        ctx.save_for_backward(x, m, mu, v, sigma, y)
        
        return y
        
    @staticmethod
    def backward(ctx, dldy):
        x, m, mu, v, sigma, y = ctx.saved_tensors
        B, H = x.shape
        
        dldgamma = torch.einsum('bq,bq,b->q', [dldy, mu, 1/sigma])
        dldbeta = dldy.sum(axis=0)
        
        dldx = (
            dldy*gamma.unsqueeze(0) / sigma.unsqueeze(1)
            - 1/H * torch.einsum('ph,h,p->p', [dldy, gamma, 1/sigma]).unsqueeze(1)
            - 1/H * mu * torch.einsum('ph,h,ph,p->p', [dldy, gamma, mu, sigma**(-3)]).unsqueeze(1)
        )
        
        return dldx, dldgamma, dldbeta
```

PyTorch also provides a function called [gradcheck](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html) to calculate the gradient of a layer using finite-differences and check to see if the backward function matches. So, we can also use that to assert the layer is correct:

```python
torch.manual_seed(42)
B, H = 32, 64
x = torch.rand((B, H), dtype=torch.float64, requires_grad=True)
gamma = torch.rand(H, dtype=torch.float64, requires_grad=True)
beta = torch.rand(H, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(LayerNormManual.apply, (x,gamma,beta), eps=1e-6, atol=0.1, rtol=0.1)
```

Notice that the input tensor dtypes have been increased to float64. The grad check fails when using float32, likely due to the numerical instability of the check.

## Next

Further examples of calculating gradients using tensor calculus and index notation can be found on the [intro page]({% link _tabs/gradients for backpropagation.md %}).
