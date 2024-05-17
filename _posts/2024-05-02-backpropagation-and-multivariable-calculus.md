---
title: "Backpropagation and Multivariable Calculus"
description: >-
  A quick intro on backpropagation and multivariable calculus for deep learning
#author: Josh Levy-Kramer
date: 2024-05-02 12:01:00 +0000
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

This post offers a concise overview of multivariable calculus and backpropagation to help you derive the gradients necessary for backpropigation in deep learning. This is a whistle-stop tour. For a more in-depth description, refer to the excellent article [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/). The [next post](/posts/the-tensor-calculus-you-need-for-deep-learning/) demonstrates how to combine these techniques with tensor calculus to derive gradients for any tensor function.

## Backpropagation

In deep learning, when training using batches of data, the model's weights are adjusted based on the calculated loss $l$. To be able to update the weights we first need to calculate the gradient of the loss $l$ with respect to all weights of the model, which tells us how to adjust the weights to reduce the loss for the current batch of data. Auto-differentiation or backpropagation is the most popular algorithm for calculating such gradients.

Taking PyTorch as an example, one would execute the [`backward()`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html) function on the loss to determine the gradients relative to variables. PyTorch accomplishes this by tracking each operation (forward function) that contributes to the loss calculation. Every forward function has a corresponding backward function, and these backward functions are run in reverse order to the forward functions to compute the gradients.

Frameworks like PyTorch have predefined backward functions that typically suffice for most users. However, for those seeking a deeper understanding or needing to implement a custom operation, it's necessary to understand how to define and manipulate these backward functions.

In general, for any function $f$ that takes $M$ inputs ($x$'s) and produces $N$ outputs ($y$'s):

$$
f: x_{1}, x_{2}, \ldots, x_{M} \mapsto y_{1}, y_{2}, \ldots, y_{N}
$$

Then there is an associated "backward" function $g$, which takes $M+N$ inputs and produces $M$ outputs:

$$
g: x_{1}, x_{2}, \ldots, x_{M}, \frac{\partial l}{\partial y_{1}}, \frac{\partial l}{\partial y_{2}}, \ldots, \frac{\partial l}{\partial y_{N}} \mapsto \frac{\partial l}{\partial x_{1}}, \frac{\partial l}{\partial x_{2}}, \ldots, \frac{\partial l}{\partial x_{M}}
$$

The inputs are the input of $f$ and the gradient of the loss $l$ with respect to each output of $f$. And the outputs are the gradient of the loss $l$ with respect to each input of $f$.

The backward function $g$, also known as the vector-Jocobian product (VJP), calculates the backpropagated gradients. More on this later. For example, in PyTorch, you can define an auto-differentiable operation by providing a forward and backward function pair like so (by implementing [`torch.autograd.Function`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)):

```python
import torch
class MatrixMultiplication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        y= x @w
        return y
    @staticmethod
    def backward(ctx, dldy: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x, w = ctx.saved_tensors
        dldx = dldy @ w.T
        dldw = x.T @ dldy
        return dldx, dldw
```

Don't worry about how the gradients are calculated - we will get onto that later. Importantly, each input of the forward function has a corresponding gradient output in the backward function, and each output of the forward function has a corresponding input of the backward function. The shapes of the tensors should also correspond with each other, e.g. `x.shape == dldx.shape`, `w.shape == dldw.shape` and `y.shape == dldy.shape`. The forward inputs are also captured using the `ctx.save_for_backward` function and obtained in the backwards function using the `ctx.saved_tensors` class attribute.

## Multivariable calculus

Most functions in deep learning have multiple inputs and outputs. A function that has multiple scalar inputs $f(u, v)$ is equivalent to a function that has a vector input $f(\hat{x})$, where $u$ and $v$ are the components of $\hat{x}$. This is known as a multivariable function, and it is said that "$f$ takes a vector" to indicate it has a vector input.

If a function has multiple scalar outputs, these can also be collected into a vector, e.g. $f: x \mapsto y_{1}, y_{2}$ is equivalent to a function with a vector output $f: x \mapsto \hat{y}$ and is said to be multivalued or vector-valued. Another way of looking at it is that a multivalued function is equivalent to multiple functions stacked into a vector $\hat{y}=\left(y_{0}, y_{1}\right)^{T}$.

This article is concerned with multivariable multivalued functions, but as that's such a mouthful, we will call them multivariable functions.

If we want to calculate the derivative of a multivariable function, we need to consider the gradient of each output with respect to each input. If a function $f$ has $M$ inputs ($x$) and $N$ outputs ($y$), there are $M*N$ gradients associated with the function, which can be denoted as $\partial y_{i} / \partial x_{j}$. Importantly $i$ and $j$ should enumerate over all scalar inputs and outputs, respectively. It doesn't matter if the scalars are arranged into a vector, matrix or tensor - the process remains the same.

A useful tool in calculus is the chain rule. Let $y$ be the output of a function which takes $M$ scalar inputs $u_{1}, \ldots, u_{M}$ which all depend on $x$, the derivative of $y$ with respect to $x$ can be shown to be:

$$
\frac{\partial y\left(u_{1}, \ldots, u_{M}\right)}{\partial x}=\sum_{i=1}^{M} \frac{\partial y}{\partial u_{i}} \frac{\partial u_{i}}{\partial x}
$$

In other words, the derivative of $y$ with respect to $x$ is the weighted sum of all $x$ contributions to the change in $y$. It is assumed that $y$ is not directly a function of $x$, it's only a function of $x$ through the intermediate functions. For example, $y=u_{1}(x) * u_{2}(x)+x^{2}$ isn't valid, you need to substitute the last term so it's $y=u_{1}(x) * u_{2}(x)+u_{3}(x) ; u_{3}=x^{2}$.

The $u$ variables could be part of data structures such as a vector, matrix or tensor. The above chain rule considers the constituent scalars and the data structures do not affect the rule. For example, given the function $y(\hat{u})$ and $\hat{u}=\left(u_{0}, \ldots, u_{M}\right)^{T}$, the chain rule above is still valid.

If the function has multiple data structures as inputs, we must consider all inputs in the chain rule. For example, given the function $y(\hat{a}, \hat{b})$, the chain rule would be ($i$ iterates over the full length of both vectors):

$$
\frac{\partial y(\hat{a}, \hat{b})}{\partial x}=\sum_{i} \frac{\partial y}{\partial a_{i}} \frac{\partial a_{i}}{\partial x}+\sum_{j} \frac{\partial y}{\partial b_{j}} \frac{\partial b_{j}}{\partial x}
$$

Throughout this article, we use the partial derivative notation $\partial y / \partial x$ instead of the "normal" derivative notation $\mathrm{d} y / \mathrm{d} x$. This is because the functions are always assumed to be multivariable, and we don't know the relationship between those variables without further context. Partial derivatives can be reinterpreted as normal derivatives with further context.

If the chain rule above can be applied to functions that use any data structure, why do we need vector, matrix or tensor calculus? Because using matrices and tensors can greatly simplify the algebra.

In vector calculus, when a function takes a vector $x$ of length $M$ and is vector-valued with an output $y$ of length $N$, the derivatives $\partial y_{i} / \partial x_{j}$ can be arranged into a $M$ by $N$ matrix known as the Jacobian matrix or just Jacobian:

$$
\frac{\partial \hat{y}}{\partial \hat{x}}=\left(\begin{array}{ccc}
\frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{N}} \\
\vdots & & \vdots \\
\frac{\partial y_{M}}{\partial x_{1}} & \cdots & \frac{\partial y_{M}}{\partial x_{N}}
\end{array}\right)
$$

Note that above, we are using what's known as a numerator layout whereby the inputs are enumerated horizontally, and the outputs are vertically. Confusingly, other authors might use denominator layout or mixed-layout conventions, which enumerate them differently. Some authors also add a transpose to the vector in the denominator e.g. $\frac{\partial \hat{y}}{\partial \hat{x}^{T}}$. This is purely notational and indicates that the $x$ inputs go horizontal in the Jacobian. Essentially, if you are using a consistent layout, then $\frac{\partial \hat{y}}{\partial \hat{x}}=\frac{\partial \hat{y}}{\partial \hat{x}^{T}}$.

The Jacobian provides a convenient data structure to group the gradients. It simplifies the application of the chain rule by utilizing matrix multiplication, which aggregates all contributions of $x$ to each output of $y$ through a weighted sum:

$$
\frac{\partial \hat{y}(\hat{u})}{\partial \hat{x}}=\frac{\partial \hat{y}}{\partial \hat{u}} \frac{\partial \hat{u}}{\partial \hat{x}}
$$

If the chain rule we first introduced can be applied to functions that use any data structure, why do we need vector, matrix or tensor calculus? As we have seen, using matrices can significantly streamline the algebra involved.

It seems logical to discuss matrix functions next; however, since the derivative of one matrix with respect to another is a tensor, we must first introduce tensors to proceed further. Following the examples, the next section will explore the application of multivariable calculus to backpropagation, and then [Part 2 introduce tensors and tensor calculus](/posts/the-tensor-calculus-you-need-for-deep-learning/).

### Example: sum

Let's consider a "sum" operator which adds together two variables:

$$
y\left(x_{1}, x_{2}\right)=x_{1}+x_{2}
$$

This is the same as:

$$
\begin{aligned}
\hat{x} & =\binom{x_{1}}{x_{2}} \\
y(\hat{x}) & =x_{1}+x_{2}
\end{aligned}
$$

To calculate the gradient of $y$ with respect to $\hat{x}$:

$$
\frac{\partial y}{\partial \hat{x}}=\left(\begin{array}{ll}
\frac{\partial y}{\partial x_{1}} & \frac{\partial y}{\partial x_{2}}
\end{array}\right)=\left(\begin{array}{ll}
1 & 1
\end{array}\right)
$$

### Example: broadcast

For another example, let's consider two different functions that copy a value $x$:

$$
\begin{aligned}
& y_{1}(x)=x \\
& y_{2}(x)=x
\end{aligned}
$$

We can then collect these functions into an array:

$$
\hat{y}(x)=\binom{x}{x}
$$

To calculate the gradient of $\hat{y}$ with respect to $x$:

$$
\frac{\partial \hat{y}}{\partial x}=\binom{\frac{\partial y_{1}}{\partial x}}{\frac{\partial y_{2}}{\partial x}}=\binom{1}{1}
$$

## Application to backpropagation

Here, we apply multivariable calculus to derive the backward graph. For example, provided with a layer in a model which executes:

$$
\begin{array}{ll}
u(\hat{a}) & =\hat{a} \cdot \hat{a} \\
v(\hat{a}, b) & =b|\hat{a}| \\
y(u) & =u+v \\
z(u, v) & =2 u
\end{array}
$$

We can visually represent the dependencies, known as the forward graph:

![Forward graph](/assets/img/backprop_1.svg){: width="150" }

The forward graph represents many sub-graphs for each input and output combination. For example, if we consider the dependency of $y$ on $\hat{a}$, we would obtain the sub-graph:

![sub-graph](/assets/img/backprop_2.svg){: width="150" }

Different input and output combinations use the same forward functions and allow us to reuse computation, e.g. we only compute $u$ once as it can be reused when computing $y$ and $z$.

To obtain the backward graph, first, we assume that a downstream scalar function $l$ consumes all outputs, so in our case $l(y, z)$. We don't need to know the function, just that the outputs are consumed by $l$. So, let's redraw the forward graph with the single output $l$.

![forward-graph-with-loss](/assets/img/backprop_3.svg){: width="150" }

We also assume we are provided with backpropagated gradients for those outputs, $\partial l / \partial y$ and $\partial l / \partial z$. Then, we need to apply the chain rule for each function to find the dependencies of the gradients. For example, if we focus on $u$, it is consumed by two functions $y$ and $z$, both consumed by l. So:

$$
\frac{\partial l(y(u), z(u))}{\partial u}=\frac{\partial l}{\partial y} \frac{\partial y}{\partial u}+\frac{\partial l}{\partial z} \frac{\partial z}{\partial u}
$$

This means that $\partial l / \partial u$ has a dependency on $\partial l / \partial y$ and $\partial l / \partial z$, which is the reverse dependency in the forward graph. This pattern holds for all the functions in the graph, and so to obtain the backward graph, we reverse all the dependencies. For example:

![backward-graph](/assets/img/backprop_4.svg){: width="150" }

In the diagram, we are using a dash to represent the backpropagated gradients, e.g. $y^{\prime}=\partial l / \partial y$. There is also a dashed line between $l$ and $y^{\prime}$, as we don't know the function that maps from $l$ to $y^{\prime}$; all we need is the values of the backpropagated gradients $y^{\prime}$ and $z^{\prime}$.

Again, the backward graph is many sub-graphs which map the inputs to every output. Each sub-graph can overlap, leading to computation reuse, making it efficient.

## Next

We have done a quick tour of multivarible calculus and backpropigation. Next [Part 2 introducing tensors and tensor calculus](/posts/the-tensor-calculus-you-need-for-deep-learning/).
