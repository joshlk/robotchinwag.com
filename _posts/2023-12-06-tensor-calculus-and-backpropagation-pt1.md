---
title: "Tensor calculus and backpropagation: part 1"
description: >-
  Introduction on Tensor calculus and backpropagation
#author: Josh Levy-Kramer
date: 2023-12-06 12:01:00 +0000
categories: [AI, Tensor Calculus]
tags: [ai, deep learning, maths, tensor calculus, automatic differentiation]  # TAG names should always be lowercase
pin: true
math: true
---

# [Draft] Tensor calculus and backpropagation: how to derive gradients in deep learning 

Note: math formulas in confluence are not compatible with the mobile app or dark mode. The only fix is to switch to the website and use light mode.

## Introduction

How do we calculate the gradient of a function? Specifically, how do we define the "backward" function or the differentiation rules for an operation in a deep learning framework?

Functions in deep learning frequently use matrices or higher-dimensional objects called tensors. When we derive the gradient of a matrix with respect to another matrix the result is a tensor, and the corresponding chain rule involves tensor products that are not representable using matrices. Thus, a solid understanding of tensor calculus is necessary to get a complete picture of gradients in deep learning.

Many texts on deep learning rely on vector calculus to derive gradients, which involves differentiating vectors with respect to vectors and organizing the resulting derivatives into matrices. However, this falls short of a systematic method to differentiate matrices or tensors and can confuse beginners. While matrix calculus, particularly Magnus \& Neudecker framework, does offer an alternative by flattening matrices into vectors, it doesn't effectively extend to tensors, and the algebra is bloated. Therefore, this article introduces tensor calculus as a more robust and generalized solution for deriving gradients within deep learning.

Tensor calculus is unfamiliar to most practitioners, and the available literature relevant to deep learning is scarce. Acquiring the necessary background means navigating various disciplines, often using inconsistent notation and conventions. This complexity can be a significant barrier for newcomers seeking a single source of truth.

This article addresses this gap by providing a consistent framework that links multivariable calculus, backpropagation and tensor calculus. The initial sections offer a concise overview of the necessary background in multivariable calculus and backpropagation. For a more indepth description, refer to the excellent article The Matrix Calculus You Need For Deep Learning. Then, tensors and tensor calculus are introduced, aiming to provide a complete reference with examples following each section.

## Backpropagation

In deep learning, when training using batches of data, the model's weights are adjusted based on the calculated loss $l$. To be able to update the weights we first need to calculate the gradient of the loss $l$ with respect to all weights of the model, which tells us how to adjust the weights to reduce the loss for the current batch of data. Auto-differentiation or backpropagation is the algorithm to calculate such gradients.

Taking PyTorch as an example, one would execute the backward() method on the loss to determine the gradients relative to variables. PyTorch accomplishes this by tracking each operation (forward function) contributing to the loss calculation. Every forward function has a corresponding backward function, and these backward functions are run in reverse order to the forward functions to compute the gradients.

Frameworks like PyTorch have predefined backward functions that typically suffice for most users. However, for those seeking a deeper understanding or needing to implement a custom operation, it's necessary to understand how to define and manipulate these backward functions.

In general, for any function $f$ that takes $M$ inputs ( $x$ 's) and produces $N$ outputs ( $y$ 's):

$$
f: x_{1}, x_{2}, \ldots, x_{M} \mapsto y_{1}, y_{2}, \ldots, y_{N}
$$

Then there is an associated "backward" function $g$, which takes $M+N$ inputs and produces $M$ outputs:

$$
g: x_{1}, x_{2}, \ldots, x_{M}, \frac{\partial l}{\partial y_{1}}, \frac{\partial l}{\partial y_{2}}, \ldots, \frac{\partial l}{\partial y_{N}} \mapsto \frac{\partial l}{\partial x_{1}}, \frac{\partial l}{\partial x_{2}}, \ldots, \frac{\partial l}{\partial x_{M}}
$$

The inputs are the input of $f$ and the gradient of the loss / with respect to each output of $f$, and the outputs are the gradient of the loss / with respect to each input of $f$.

The backward function $g$, also known as the vector-Jocobian product (VJP), calculates the backpropagated gradients. More on this later. For example, in PyTorch, you can define an auto-differentiable operation by providing a forward and backward function pair like so:

```
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

Don't worry about how the gradients are calculated - we will get onto that later. Importantly, each input of the forward method has a corresponding gradient output in the backward method, and each output of the forward method has a corresponding input of the backward method. The shapes of the tensors should also correspond with each other, e.g. x.shape == dldx.shape, w.shape == dldw.shape and $y$.shape == dldy.shape. The forward inputs are also captured using the ctx.save_for_backward method and obtained in the backwards method using the ctx.saved_tensors property.

## Multivariable calculus

Most functions in deep learning have multiple inputs and outputs. A function that has multiple scalar inputs $f(u, v)$ is equivalent to a function that has a vector input $f(\hat{x})$, where $u$ and $v$ are the components of $\hat{x}$. This is known as a multivariable function, and it is said that " $f$ takes a vector" to indicate it has a vector input.

If a function has multiple scalar outputs, these can also be collected into a vector, e.g. $f: x \mapsto y_{1}, y_{2}$ is equivalent to a function with a vector output $f: x \mapsto \hat{y}$ and is said to be multivalued or vector-valued. Another way of looking at it is that a multivalued function is equivalent to multiple functions stacked into a vector $\hat{y}=\left(y_{0}, y_{1}\right)^{T}$.

This article is concerned with multivariable multivalued functions, but as that's such a mouthful, we will call them multivariable functions.

If we want to calculate the derivative of a multivariable function, we need to consider the gradient of each output with respect to each input. If a function $f$ has $M$ inputs $(x)$ and $N$ outputs $(y)$, there are $M^{*} N$ gradients associated with the function, which can be denoted as $\partial y_{i} / \partial x_{j}$. Importantly $i$ and $j$ should enumerate over all scalar inputs and outputs, respectively. It doesn't matter if the scalars are arranged into a vector, matrix or tensor (more on that later) - the process remains the same.

A useful tool in calculus is the chain rule. Let $y$ be the output of a function which takes $M$ scalar inputs $u_{1}, \ldots, u_{M}$ which all depend on $x$, the derivative of $y$ with respect to $x$ can be shown to be:

$$
\frac{\partial y\left(u_{1}, \ldots, u_{M}\right)}{\partial x}=\sum_{i=1}^{M} \frac{\partial y}{\partial u_{i}} \frac{\partial u_{i}}{\partial x}
$$

In other words, the derivative of $y$ with respect to $x$ is the weighted sum of all $x$ contributions to the change in $y$. It is assumed that $y$ is not directly a function of $x$, it's only a function of $x$ through the intermediate functions. For example, $y=u_{1}(x) * u_{2}(x)+x^{2}$ isn't valid, you need to substitute the last term so it's $y=u_{1}(x) * u_{2}(x)+u_{3}(x) ; u_{3}=x^{2}$.

The $u$ variables could be part of data structures such as a vector, matrix or tensor. The above chain rule considers the constituent scalars and the data structures do not affect the rule. For example, given the function $y(\hat{u})$ and $\hat{u}=\left(u_{0}, \ldots, u_{M}\right)^{T}$, the chain rule above is still valid.

If the function has multiple data structures as inputs, we must consider all inputs in the chain rule. For example, given the function $y(\hat{a}, \hat{b})$, the chain rule would be ( $i$ iterates over the full length of both vectors):

$$
\frac{\partial y(\hat{a}, \hat{b})}{\partial x}=\sum_{i} \frac{\partial y}{\partial a_{i}} \frac{\partial a_{i}}{\partial x}+\sum_{i} \frac{\partial y}{\partial b_{i}} \frac{\partial b_{i}}{\partial x}
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

It seems logical to discuss matrix functions next; however, since the derivative of one matrix with respect to another is a tensor, we must first introduce tensors to proceed further. Following the examples, the next section will explore the application of multivariable calculus to backpropagation, and then we introduce tensors.

## Example: sum

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

To calculate the gradient of $y$ with respect to $\hat{x}$ :

$$
\frac{\partial y}{\partial \hat{x}}=\left(\begin{array}{ll}
\frac{\partial y}{\partial x_{1}} & \frac{\partial y}{\partial x_{2}}
\end{array}\right)=\left(\begin{array}{ll}
1 & 1
\end{array}\right)
$$

## Example: broadcast

For another example, let's consider two different functions that copy a value $x$ :

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

To calculate the gradient of $\hat{y}$ with respect to $x$ :

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

![](https://cdn.mathpix.com/cropped/2024_05_07_c7c3c439079236dbfba7g-04.jpg?height=307&width=203&top_left_y=1397&top_left_x=191)

The forward graph represents many sub-graphs for each input and output combination. For example, if we consider the dependency of $y$ on $\hat{a}$, we would obtain the sub-graph:

![](https://cdn.mathpix.com/cropped/2024_05_07_c7c3c439079236dbfba7g-04.jpg?height=298&width=200&top_left_y=1846&top_left_x=190)

Different input and output combinations use the same forward functions and allow us to reuse computation, e.g. we only compute $u$ once as it can be reused when computing $y$ and $z$.

To obtain the backward graph, first, we assume that a downstream scalar function $l$ consumes all outputs, so in our case $l(y, z)$. We don't need to know the function, just that the outputs are consumed by $l$. So, let's redraw the forward graph with the single output $l$.

![](https://cdn.mathpix.com/cropped/2024_05_07_c7c3c439079236dbfba7g-05.jpg?height=412&width=193&top_left_y=111&top_left_x=191)

We also assume we are provided with backpropagated gradients for those outputs, $\partial l / \partial y$ and $\partial l / \partial z$. Then, we need to apply the chain rule for each function to find the dependencies of the gradients. For example, if we focus on $u$, it is consumed by two functions $y$ and $z$, both consumed by l. So:

$$
\frac{\partial l(y(u), z(u))}{\partial u}=\frac{\partial l}{\partial y} \frac{\partial y}{\partial u}+\frac{\partial l}{\partial z} \frac{\partial z}{\partial u}
$$

This means that $\partial l / \partial u$ has a dependency on $\partial l / \partial y$ and $\partial l / \partial z$, which is the reverse dependency in the forward graph. This pattern holds for all the functions in the graph, and so to obtain the backward graph, we reverse all the dependencies. For example:

![](https://cdn.mathpix.com/cropped/2024_05_07_c7c3c439079236dbfba7g-05.jpg?height=415&width=206&top_left_y=942&top_left_x=187)

In the diagram, we are using a dash to represent the backpropagated gradients, e.g. $y^{\prime}=\partial l / \partial y$. There is also a dashed line between $l$ and $y^{\prime}$, as we don't know the function that maps from $l$ to $y^{\prime}$; all we need is the values of the backpropagated gradients $y^{\prime}$ and $z^{\prime}$.

Again, the backward graph is many sub-graphs which map the inputs to every output. Each sub-graph can overlap, leading to computation reuse, making it efficient.

## Tensors

A tensor is a multi-dimensional ordered array of numbers, expanding the concept of a matrix into N-dimensions. Here, we are specifically talking about Cartesian tensors, which simplify the broader and more complex idea of tensors typically discussed in physics or mathematics. Focusing on Cartesian tensors removes the need to discuss complex topics such as universality, dual spaces, and tensor fields, which, while important in some fields, are not directly applicable to deep learning.

A tensor $\mathcal{T}$ has components $t_{i \ldots j}$, where $i \ldots j$ means an arbitrary list of indices, including the indices $i$ and $j$. The number of indices indicates the number of axes the tensor has known as its rank or order; for example, $t_{i j}$ is an order-2 tensor. Tensors are denoted using an uppercase calligraphic font with the corresponding components in lowercase and with indices using lowercase Latin symbols.

An order-1 tensor is analogous to vectors, and an order-2 tensor is analogous to matrices. You can also have an order-0 tensor, which is a scalar or single number.

Representing three-dimensional or higher objects on paper is challenging, so we need a method to flatten their structure. One approach is to utilize basis vectors. For example, a vector can be expressed as a linear combination of basis vectors:

$$
\hat{x}=1 \hat{e}_{1}+2 \hat{e}_{2}=\binom{1}{2}
$$

Where a basis vector is defined as:

$$
\hat{e}_{1}=\binom{1}{0}
$$

The bases of a matrix can be constructed by applying the tensor product $(\otimes)$ of two basis vectors. Don't worry about the details of how the tensor product works - we won't be using it much:

$$
\hat{e}_{1} \otimes \hat{e}_{2}=\hat{e}_{1,2}=\binom{1}{0} \otimes\binom{0}{1}=\left(\begin{array}{ll}
0 & 1 \\
0 & 0
\end{array}\right)
$$

Then a matrix can be expressed using its basis:

$$
\begin{aligned}
X & =1 \hat{e}_{1} \otimes \hat{e}_{1}+2 \hat{e}_{1} \otimes \hat{e}_{2}+3 \hat{e}_{2} \otimes \hat{e}_{1}+4 \hat{e}_{2} \otimes \hat{e}_{2} \\
& =\left(\begin{array}{ll}
1 & 2 \\
3 & 4
\end{array}\right)
\end{aligned}
$$

Now, to represent an order-3 tensor, we can write it as a linear combination of its basis:

$$
\begin{aligned}
\mathcal{T}= & 1 \hat{e}_{1,1,1}+2 \hat{e}_{1,1,2}+3 \hat{e}_{1,2,1}+4 \hat{e}_{1,2,2} \\
& +5 \hat{e}_{2,1,1}+6 \hat{e}_{2,1,2}+7 \hat{e}_{2,2,1}+8 \hat{e}_{2} \hat{e}_{2,2,2} \\
= & \sum_{i, j, k} t_{i j k} \hat{e}_{i j k} \\
= & \hat{e}_{1} \otimes\left(\begin{array}{ll}
1 & 2 \\
3 & 4
\end{array}\right)+\hat{e}_{2} \otimes\left(\begin{array}{ll}
5 & 6 \\
7 & 8
\end{array}\right)
\end{aligned}
$$

As you can see, the notation gets unwieldy very quickly when the number of dimensions grows larger than 2 . We also need a way of defining operations between tensors. We will, therefore, turn to index notation to provide a more convenient way to represent tensors of any order.

## Index notation

Index notation provides a convenient algebra to work with individual elements or components instead of the tensor as a whole. For example, a matrix multiplication can be expressed as:

$$
c_{i k}=a_{i j} b_{j k}
$$

Below, we explain the rules of index notation and how to understand the above expression.

Index notation (not to be confused with multi-index notation) is a simplified version of Einstein notation or Ricci calculus that works with Cartesian tensors. Importantly, not all the normal algebra rules apply with index notation, so we must take care when first using it.

Be aware that different authors use various flavours of index notation, so when reading other texts, look at the details of how they use the notation.

## Index notation rulebook

## Tensor indices

- A tensor is written using uppercase curly font $\mathcal{T}$. Its components are written in lowercase, with its indices in subscript $t_{i j k}$. You can obtain the components of a tensor using square brackets e.g. $[\mathcal{T}]_{i j k}=t_{i j k}$.
- The number of indices represents the order of the tensor. For example $t_{i j k}$ has three indices $i, j$ and $k$, and so is an order- 3 tensor.
- The indices can use any letters, so without further context $t_{i j k}$ and $t_{a b c}$ refer to the same tensor and components.
- The range of the indices is either determined by context or can be kept arbitrary.
- Indices are labelled using lowercase Latin symbols.


## Free indices

- Free indices describe a system of equations and imply that the expression is repeated, enumerating the index range. For example, if the index $i$ has the range $i=\{1,2,3\}$, then:

$$
a_{i}=b_{i} c_{i} \Leftrightarrow\left\{\begin{array}{l}
a_{1}=b_{1} c_{1} \\
a_{2}=b_{2} c_{2} \\
a_{3}=b_{3} c_{3}
\end{array}\right.
$$

- A free index must appear in every additive term in the equation. Most of the time can determine free indices by checking an equation's left-hand side (LHS) or the subject of an equation.
- A free index can be renamed if it is renamed in all additive terms.
- An equation can have multiple free indices, representing the system of equations that enumerate all indices.


## Dummy indices

- Dummy indices appear twice or more in an additive term and do not appear on the LHS or equation subject. Dummy indices imply a summation, also known as a contraction:

$$
a=b_{i} c_{i} \Leftrightarrow a=\sum_{i} b_{i} c_{i}
$$

- A dummy index can take any label not being used by a free index.
- A dummy index is "local" to each additive term, so it can be renamed in one term and not others.
- Dummy indices and free indices can be used together. For example, below $i$ is on the LHS and $k$ is missing, which means that $i$ is a free index and $k$ is a dummy index:

$$
a_{i}=b_{i k} c_{i k} d_{k}
$$

- Sometimes, it is ambiguous if an index is a dummy or free index; on these occasions, free indices can be stated explicitly by writing "no sum". For example, below $i$ is a free index and $j$ is a dummy index:

$$
a_{i j} b_{i j}=\alpha_{i j} \beta_{i j} \quad(\text { no sum i) }
$$

## Kronecker delta

- The Kronecker delta is an order-2 tensor whose components are 1 when the indices equal and zero otherwise:

$$
\delta_{i j}= \begin{cases}1 & \text { if } i=j \\ 0 & \text { otherwise }\end{cases}
$$

- The Kronecker delta is analogous to the identity matrix in matrix notation.
- The tensor is symmetric $\delta_{i j}=\delta_{j i}$.
- The Kronecker delta can be used to select a specific term in a summation. For example, we can select the $k^{\prime}$ 'th term from a contraction of index $i$ of $a$ :

$$
\delta_{k i} a_{i j}=\delta_{k 0} a_{0 j}+\delta_{k 1} a_{1 j}+\ldots=\delta_{k k} a_{k j}=a_{k j}
$$

- More generally, if index $i$ is a dummy index and $k$ is a dummy or free index, using the Kronecker delta $\delta_{i k}$, we can replace index $i$ with all instances of index $k$. Consider these separate examples:

$$
\begin{array}{rlr}
a_{j} & =\delta_{i k} b_{i j}=b_{k j} & \text { (replace i with k) } \\
a_{j} & =\delta_{k i} b_{i j} c_{i j}=b_{k j} c_{k j} & \text { (replace all i with k) } \\
a_{p q} & =\delta_{i j} b_{i p} c_{q j}=b_{j p} c_{q j}=b_{i p} c_{q i} & \text { (replace i with j, or j with i) } \tag{replacealliwithk}
\end{array}
$$

## One-tensor

- The one-tensor is a tensor with the value 1 for all its components. It can be of any arbitrary order or shape:

$$
\mathbf{1}_{i \ldots j}=1
$$

- The one-tensor can be used to contract components of another tensor, for example, summing all components of $b$ :

$$
a=\mathbf{1}_{i} b_{i}
$$

- The one-tensor can be used to broadcast components of another tensor, for example, broadcasting an order-1 tensor into an order-2 tensor (the values of $b$ are repeated for all values of $i$ ):

$$
a_{i j}=\mathbf{1}_{i} b_{j}
$$

- The contraction of a one-tensor with itself equals the cardinality of an index. For example, if $n$ has a range between 1 and $N$, then:

$$
a=\mathbf{1}_{n} \mathbf{1}_{n}=N
$$

This is true for any number of one-tensors being contracted with each other:

$$
a=\mathbf{1}_{n} \mathbf{1}_{n} \ldots \mathbf{1}_{n}=N
$$

- The broadcast of an order- $M$ one-tensor with another order- $N$ one-tensor is equal to an order- $(M+N)$ one-tensor:

$$
a_{i j}=\mathbf{1}_{i} \mathbf{1}_{j}=\mathbf{1}_{i j}
$$

- A one-tensor can be dropped if a term already has a tensor with the same free or dummy index as the one-tensor, as this results in multiplying the terms by 1 :

$$
a_{i}=\mathbf{1}_{i j} b_{i j} c_{j}=b_{i j} c_{j}
$$

## Substitution

When substituting one equation into another...

- You need to use the same free indices. For example, given $a_{i}=b_{i} c_{i}$ and $c_{j}=d_{j j}$, it would be wrong to conclude then $a_{i}=b_{i} d_{j j}$. Instead, we must change the free indices so they match to get $a_{i}=b_{i} d_{i i}$.
- You need to use different dummy indices. For example, given $a_{i}=b_{i j} \mathbf{1}_{j} c_{i}$ and $c_{i}=d_{i j} e_{j}$, we need to change $j$ to a new index $k$ in one of the equations to get $a_{i}=b_{i j} \mathbf{1}_{j} d_{i k} e_{k}$, otherwise we get $a_{i}=b_{i j} \mathbf{1}_{j} d_{i j} e_{j}$ which is wrong.
- You cannot substitute into a contraction. For example, given $a=b_{k} c_{k} d_{k}$ and $\beta=c_{k} d_{k}$ it would be wrong to conclude $a=b_{k} \beta$ because $b_{k}$ is also part of the contraction.


## Properties of addition and multiplication

- Only tensors of the same order can be added together, for example, $a_{i j}+b_{i j}$ is valid while $a_{i j}+b_{i}$ is not. Except for adding a tensor with a scalar $a_{i j}+b$, which is implicitly broadcasted to the appropriate order $a_{i j}+b \mathbf{1}_{i j}$.
- The usual commutative and associative properties of addition and multiplications remain true (see appendix).
- Tensors don't distribute or factorise in the usual way. An expression can only be factorised, provided that all additive terms result in having the same order. For example:

$$
\begin{aligned}
a_{i} & =b_{i j} c_{i j}+b_{i j} d_{j} \\
& \neq b_{i j}\left(c_{i j}+d_{j}\right)
\end{aligned}
$$

But a one-tensor can be used to maintain the order of the terms:

$$
\begin{aligned}
a_{i} & =b_{i j} c_{i j}+b_{i j} d_{j} \\
& =b_{i j}\left(c_{i j}+\mathbf{1}_{i} d_{j}\right)
\end{aligned}
$$

## Algebra of index notation

- Provided an equation, it is valid to apply a function on both sides:

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
f\left(a_{i}\right) & =f\left(b_{i k} c_{k}\right)
\end{aligned}
$$

- This includes adding a tensor to both sides, for example adding $\gamma_{i}$ :

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
a_{i}+\gamma_{i} & =b_{i k} c_{k}+\gamma_{i}
\end{aligned}
$$

- Or multiplying both sides with a scalar $\lambda$ :

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
a_{i} \lambda & =b_{i k} c_{k} \lambda
\end{aligned}
$$

- Or multiplying both sides using new free indices $\gamma_{j}$ :

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
a_{i} \gamma_{j} & =b_{i k} c_{k} \gamma_{j}
\end{aligned}
$$

- Or multiply both sides with an existing free index to contract both sides $\gamma_{i}$ :

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
a_{i} \gamma_{i} & =b_{i k} c_{k} \gamma_{i}
\end{aligned}
$$

- It is not valid to contract an index that is already a dummy index, for example multiplying $\gamma_{k}$ :

$$
\begin{gathered}
a_{i}=b_{i k} c_{k} \\
a_{i} \gamma_{k} \neq b_{i k} c_{k} \gamma_{k}
\end{gathered}
$$

To show this further, consider another example where $a_{k}=(2,-1)^{T} ; b_{k}=(0,1)^{T} ; \gamma=(0,1)^{T}$ :

$$
\begin{aligned}
a_{k} \mathbf{1}_{k} & =b_{k} \mathbf{1}_{k} \\
a_{k} \mathbf{1}_{k} \gamma_{k} & \neq b_{k} \mathbf{1}_{k} \gamma_{k} \\
2 * 0-1 * 1 & \neq 0 * 0+1 * 1 \\
-1 & \neq 1
\end{aligned}
$$

- It is valid to use existing function identities, such as logarithmic or exponential identities e.g. $\log (a b)=\log (a)+\log (b)$ or e.g. $(a b)^{2}=a^{2} b^{2}$, but they should not change the components of a contraction. For example if $a_{i j}=b_{i} c_{j}$, then $a_{i j}^{2}=b_{i}^{2} c_{j}^{2}$ is valid; but if $a_{i j}=b_{i k} c_{j k}$, then $a_{i j}^{2}=b_{i k}^{2} c_{j k}^{2}$ is not valid.

| Operation | Matrix notation | Index notation |
| :---: | :---: | :---: |
| Matrix addition | $A=B+C$ | $a_{i j}=b_{i j}+c_{i j}$ |
| Matrix transpose | $A=B^{T}$ | $a_{i j}=b_{j i}$ |
| Matrix trace | $\lambda=\operatorname{tr}(A)$ | $\lambda=a_{i i}$ |
| Scalar-matrix multiplication | $A=\lambda B$ | $a_{i j}=\lambda b_{i j}$ |
| Matrix multiplication | $A=B C$ | $a_{i j}=b_{i k} c_{k j}$ |
| Matrix Hadamard multiplication <br> (element-wise multiplication) | $A=B \odot C$ | $a_{i j}=b_{i j} c_{i j}$ |
| Vector outer product | $A=\hat{b} \hat{c}^{T}$ | $a_{i j}=b_{i} c_{j}$ |
| Vector dot product | $\lambda=\hat{b} \cdot \hat{c}$ | $\lambda=b_{i} c_{i}$ |
| Vector Euclidean norm (L2 norm) | $\lambda=\\|\hat{x}\\|$ | $\lambda=\sqrt{x_{i} x_{i}}$ |

## Summary

We now have a way of representing order-N tensors using a convenient notation. It also allows us to define new operations between arbitrary ordered tensors, which was previously difficult to do using matrix notation without inventing new symbols.

Often, instead of referring to $a_{i j}$ as the components of an order-2 tensor, we will simply refer to $a_{i j}$ as the tensor. However, it should always be remembered that $a_{i j}$ are components and not the data structure itself.

## Matrix expressions in index notation

Tensors and matrix notation are analogous when used to represent the same set of equations. This similarity arises from their ability to depict comparable data structures and the operations between them.

Here are some common matrix operations in index notation.

## Example: layer normalisation using index notation

PyTorch defines the layer normalization (layer norm) operation for an input matrix $X$, with shape batch size $(B)$ by hidden size $(H)$, as:

$$
y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta
$$

Where the mean $\mathrm{E}[x]$ and variance $\operatorname{Var}[x]$ are calculated for each sample in a batch, and $\gamma$ and $\beta$ are learnable vector weights with lengths equal to the hidden size. $\epsilon$ is a constant usually equal to $1 \mathrm{e}-05$.

Let's first look at the mean $\mathrm{E}[x]$, which is defined as:

$$
\mathrm{E}[x]=\frac{1}{H} \sum_{i=1}^{H} x_{i}
$$

The mean is calculated for each sample, so $i$ ranges over all $x$ values for a sample. With index notation, we can use the one-tensor to sum the components of $x$, drop the summation sign and explicitly state we have a value per batch by using a tensor $m_{b}$ to store the values of the mean:

$$
m_{b}=\frac{1}{H} \mathbf{1}_{h} x_{b h}
$$

$b$ is a free index which ranges from 1 to $B$, and $h$ is a dummy index ranging from 1 to $H$. Similarly, we can apply the same idea to the variance, which is defined as:

$$
\operatorname{Var}[x]=\frac{1}{H} \sum_{i=1}^{H}\left(x_{i}-\mathrm{E}[x]\right)^{2}
$$

We can represent this using index notation:

$$
v_{b}=\frac{1}{H} \mathbf{1}_{h}\left(x_{b h}-\mathbf{1}_{h} m_{b}\right)^{2}
$$

Notice how we use the one-tensor for two different purposes here. The first is used to sum the terms like we used it for the mean. The second is used to broadcast the mean $m_{b}$ to an order-2 tensor $\mathbf{1}_{h} m_{b}$, as the mean is invariant to the hidden dimension, and we want to subtract it from the order-2 tensor $x_{b h}$.

Now, we move to the definition of $\mathrm{y} . \gamma$ and $\beta$ are learnable vector weights with length equal to the hidden size, which means they are an order-1 tensor with length $H$. Using index notation, we can represent $y$ as:

$$
y_{b h}=\frac{x_{b h}-\mathbf{1}_{h} m_{b}}{\sqrt{v_{b}+\epsilon}} \gamma_{h}+\mathbf{1}_{b} \beta_{h}
$$

We use the one-tensor twice to broadcast $m_{b}$ and $\beta_{h}$ to an order-2 tensors.

## Tensor calculus

Tensor calculus allows us to consider how to differentiate a tensor with respect to a tensor.

Provided a function $\mathcal{Y}=\mathcal{F}(\mathcal{X})$, which is tensor-valued with order-M and output components $y_{i \ldots j}$, and takes a tensor $\mathcal{X}$ with order- $N$ and components $x_{p \ldots q}$, the derivative or Jacobian tensor is then of order- $(M+N)$ :

$$
\frac{\partial \mathcal{Y}}{\partial \mathcal{X}}=\left[\frac{\partial \mathcal{Y}}{\partial \mathcal{X}}\right]_{i \ldots j p \ldots q} \hat{e}_{i \ldots j p \ldots q}=\frac{\partial y_{i \ldots j}}{\partial x_{p \ldots q}} \hat{e}_{i \ldots j p \ldots q}
$$

We are going to focus on index notation, so the derivative of a tensor with respect to a tensor is the quantity $\frac{\partial y_{i \ldots j}}{\partial x_{p \ldots q}}$ and $\frac{\partial}{\partial x_{p \ldots q}}$ is the tensor derivative operator. Below, we list the rules for applying the operator, and then we can get onto some examples.

## Tensor calculus rulebook

## Tensor derivative operator

- The derivative operator should always use new free indices. For example, given the equation:

$$
y_{i . . j}=x_{i \ldots j}
$$

We pick the new free indices $a \ldots b$ when applying the derivative operator:

$$
\frac{\partial y_{i \ldots j}}{\partial x_{a \ldots b}}=\frac{\partial x_{i \ldots j}}{\partial x_{a \ldots b}}
$$

- The derivative of a tensor with itself is a product of Krockener deltas. For example, for an order-1 tensor:

$$
\frac{\partial x_{i}}{\partial x_{p}}=\delta_{i p}
$$

This is because when $i \neq p, x_{i}$ is constant to $x_{p}$. In general, for the tensor of order- $N$ :

$$
\frac{\partial x_{i . . . j}}{\partial x_{p \ldots .}}=\underbrace{\delta_{i p} \ldots \delta_{j q}}_{N}
$$

## Product and quotient rule

- We can use the product rule to obtain the tensor derivative of a product:

$$
\frac{\partial u_{a b} v_{c d}}{\partial x_{p q}}=\frac{\partial u_{a b}}{\partial x_{p q}} v_{c d}+u_{a b} \frac{\partial v_{c d}}{\partial x_{p q}}
$$

This can also be applied to a contraction ( $b$ is a dummy index):

$$
\frac{\partial u_{a b} v_{b c}}{\partial x_{p q}}=\frac{\partial u_{a b}}{\partial x_{p q}} v_{b c}+u_{a b} \frac{\partial v_{b c}}{\partial x_{p q}}
$$

And the quotient rule for division:

$$
\frac{\partial}{\partial x_{p q}}\left(\frac{u_{a b}}{v_{c d}}\right)=\frac{1}{v_{c d}^{2}}\left(\frac{\partial u_{a b}}{\partial x_{p q}} v_{c d}-u_{a b} \frac{\partial v_{c d}}{\partial x_{p q}}\right)
$$

This also works for contractions ( $u$ and $v$ can include dummy indices).

## Chain rule

- We can also use the chain rule. For example, if a function outputs $y$ and takes $u$ as input and $\mathrm{u}$ is a function of $x, y_{a \ldots b}\left(u_{c \ldots d}\left(x_{i \ldots j}\right)\right)$, then:

$$
\frac{\partial y_{a \ldots b}}{\partial x_{p \ldots q}}=\frac{\partial y_{a \ldots b}}{\partial u_{c \ldots d}} \frac{\partial u_{c \ldots d}}{\partial x_{p \ldots q}}
$$

Importantly, the two derivatives are contracted, i.e. $u_{c \ldots d}$ have the same indices in the first and second derivative terms. This mimics the summation shown in the multivariable calculus section, providing the weighted sum of all $x$ contributions to the change in $y$. Also, note we apply the derivative operator with new free indices $x_{p \ldots q}$.

- For the more general case of $n$ arbitrary nested functions $y_{a \ldots b}\left(u_{c \ldots d}\left(\ldots v_{e \ldots f}\left(x_{i \ldots j}\right)\right)\right)$, the chain rule is:

$$
\frac{\partial y_{a \ldots b}}{\partial x_{p \ldots q}}=\frac{\partial y_{a \ldots b}}{\partial u_{c \ldots d}} \frac{\partial u_{c \ldots d}}{\partial v_{e \ldots f}} \frac{\partial v_{e \ldots f}}{\partial x_{p \ldots q}}
$$

Importantly, the two contractions in the expression use different dummy indices and are separate.

- If a function takes multiple tensors as input, then this must be considered when applying the chain rule. For example, provided with the function that takes two tensors $y_{a \ldots b}\left(u_{c \ldots d}\left(x_{i \ldots j}\right), v_{e \ldots f}\left(x_{i \ldots j}\right)\right)$, the chain rule would be:

$$
\frac{\partial y_{a \ldots b}}{\partial x_{p \ldots q}}=\frac{\partial y_{a \ldots b}}{\partial u_{c \ldots d}} \frac{\partial u_{c \ldots d}}{\partial x_{p \ldots q}}+\frac{\partial y_{a \ldots b}}{\partial v_{e \ldots f}} \frac{\partial v_{e \ldots f}}{\partial x_{p \ldots q}}
$$

## Summary

We are now at the point where we can combine tensor calculus with backpropagation! We have learned how to compute derivatives and apply the chain rule to matrices and tensors of any order.

To derive the backpropagated gradients of a tensor function, the steps are A) translating the function into index notation, B) calculating the derivative tensor of the function with respect to all its inputs, C) assume the outputs are a dependency of a downstream scalar function $l$ and we are provided with the gradients of $l$ with respect to each of the outputs, and D) use the chain rule to determine the gradients of $l$ with respect to each of the inputs of the function.

Great! With this foundation, we're ready to move on to some practical examples.

## Example: element-wise functions

An element-wise or pointwise function applies an univariable function to all tensor components. For example, we could apply the trigonometric function sin to all tensor values. Let's consider an arbitrary pointwise function point:

$$
y_{i}=\operatorname{point}\left(x_{i}\right)
$$

Above $x_{i}$ and $y_{i}$ are order-1 tensors, but the following maths would apply to an arbitrary order- $N$ tensor.

Firstly, to calculate the derivative of $y_{i}$ with respect to $x_{p}$. Importantly, we differentiate using a new free index $p$ :

$$
\begin{aligned}
\frac{\partial y_{i}}{\partial x_{p}} & =\frac{\partial \operatorname{point}\left(x_{i}\right)}{\partial x_{p}} \\
& =\frac{\partial \operatorname{point}\left(x_{i}\right)}{\partial x_{i}} \frac{\partial x_{i}}{\partial x_{p}} \\
& =\operatorname{point}^{\prime}\left(x_{i}\right) \delta_{i p}
\end{aligned}
$$

Where point ${ }^{\prime}$ is the derivative of the pointwise function, for example, if the pointwise function is sin, then its derivative is cos.

Secondly, we assume $y_{i}$ is a dependency of a downstream scalar function $l$ and that we are provided with the gradient $\partial l / \partial y_{i}$. We then use the chain rule to derive the backpropagated gradient of the input $\partial l / \partial x_{p}$ :

$$
\begin{aligned}
\frac{\partial l}{\partial x_{p}} & =\frac{\partial l}{\partial y_{i}} \frac{\partial y_{i}}{\partial x_{p}} \\
\frac{\partial l}{\partial x_{p}} & =\frac{\partial l}{\partial y_{i}} \operatorname{point}^{\prime}\left(x_{i}\right) \delta_{i p} \\
& =\frac{\partial l}{\partial y_{p}} \operatorname{point}^{\prime}\left(x_{p}\right)
\end{aligned}
$$

## Example: matrix multiplication

Given $Y=X A$, we first need to convert it to index notation:

$$
y_{i j}=x_{i k} a_{k j}
$$

We have two gradients to derive as the function has two matrix inputs: $X$ and $A$.

## Gradient of $X$

Let's first obtain the derivative with respect to $X$ :

$$
\begin{aligned}
\frac{\partial y_{i j}}{\partial x_{p q}} & =\frac{\partial\left(x_{i k} a_{k j}\right)}{\partial x_{p q}} \\
& =\frac{\partial x_{i k}}{\partial x_{p q}} a_{k j} \\
& =\delta_{i p} \delta_{k q} a_{k j} \\
& =\delta_{i p} a_{q j}
\end{aligned}
$$

Then, we obtain the backpropagated gradient using the chain rule:

$$
\begin{aligned}
\frac{\partial l}{\partial x_{p q}} & =\frac{\partial l}{\partial y_{i j}} \frac{\partial y_{i j}}{\partial x_{p q}} \\
& =\frac{\partial l}{\partial y_{i j}} \delta_{i p} a_{q j} \\
& =\frac{\partial l}{\partial y_{p j}} a_{q j}
\end{aligned}
$$

We can then covert this back to matrix notation:

$$
\begin{aligned}
{\left[\frac{\partial l}{\partial X}\right]_{p q} } & =\left[\frac{\partial l}{\partial Y}\right]_{p j}\left[A^{T}\right]_{j q} \\
\frac{\partial l}{\partial X} & =\frac{\partial l}{\partial Y} A^{T}
\end{aligned}
$$

Note that we have to transpose the matrix $A$ so the indices of the two expressions resemble a matrix multiplication.

## Gradient of A

First, obtain the derivative with respect to $A$ :

$$
\begin{aligned}
\frac{\partial y_{i j}}{\partial a_{p q}} & =\frac{\partial x_{i k} a_{k j}}{\partial a_{p q}} \\
& =x_{i k} \frac{\partial a_{k j}}{\partial a_{p q}} \\
& =x_{i k} \delta_{k p} \delta_{j q} \\
& =x_{i p} \delta_{j q}
\end{aligned}
$$

Then, we obtain the backpropagated gradient:

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
\frac{\partial l}{\partial A} & =X^{T} \frac{\partial l}{\partial Y}
\end{aligned}
$$

## Example: matrix inverse

Given $Y=X^{-1}$, we know from the definition of an inverse matrix that $I=X X^{-1}=X Y$ (assuming the inverse of $X$ exists). We convert this to index notation:

$$
\delta_{i j}=x_{i k} y_{k j}
$$

First, we find the derivative with respect to $X$ and use the product rule:

$$
\begin{align*}
\frac{\partial \delta_{i j}}{\partial x_{p q}} & =\frac{\partial x_{i k} y_{k j}}{\partial x_{p q}}  \tag{nosumi,j,p,q}\\
0 & =\frac{\partial x_{i k}}{\partial x_{p q}} y_{k j}+x_{i k} \frac{\partial y_{k j}}{\partial x_{p q}} \\
0 & =\delta_{i p} \delta_{k q} y_{k j}+x_{i k} \frac{\partial y_{k j}}{\partial x_{p q}} \\
0 & =\delta_{i p} y_{q j}+x_{i k} \frac{\partial y_{k j}}{\partial x_{p q}} \\
x_{i k} \frac{\partial y_{k j}}{\partial x_{p q}} & =-\delta_{i p} y_{q j}
\end{align*}
$$

We then multiply by the inverse matrix $y_{n i}$, we must use a new free index $n$ for the first axis and contract on the second axis $i$ :

$$
\begin{aligned}
y_{n i} x_{i k} \frac{\partial y_{k j}}{\partial x_{p q}} & =-y_{n i} \delta_{i p} y_{q j} \\
\delta_{n k} \frac{\partial y_{k j}}{\partial x_{p q}} & =-y_{n p} y_{q j} \\
\frac{\partial y_{n j}}{\partial x_{p q}} & =-y_{n p} y_{q j}
\end{aligned}
$$

Secondly, we derive the backpropagated gradient using the chain rule:

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

## Example: cross-entropy loss

The cross-entropy loss is the softmax followed by the negative log-likelihood loss. Provided an input vector $\hat{x}$ of logits:

$$
\begin{aligned}
s_{i} & =\frac{e^{x_{i}}}{\mathbf{1}_{j} e^{x_{j}}} \\
c & =-\log s_{T}
\end{aligned}
$$

Where $s_{i}$ is a vector of softmax values, and $c$ is the scalar cross-entropy loss. $T$ is the index corresponding to the target label. Note that $T$ is a constant and not a free or dummy index.

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

Notice that we can drop $\mathbf{1}_{n}$ as $n$ is a free index, and $\mathbf{1}_{n}$ equals 1 .

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

It's interesting to note that because of the influence of normalizing all values by $\sigma$, all logits have a non-zero gradient even if they do not correspond to the true label.

Then, deriving the backpropagated gradient is trivial:

$$
\begin{aligned}
\frac{\partial l}{\partial x_{n}} & =\frac{\partial l}{\partial c} \frac{\partial c}{\partial x_{n}} \\
& =\frac{\partial l}{\partial c}\left(s_{n}-\delta_{T_{n}}\right)
\end{aligned}
$$

It might be the case that we start backpropagation using the cross-entropy loss, in that case $l=c$ and $\partial l / \partial c=1$.

## Example: layer normalization

This example is quite long and involved but combines the different concepts presented in this article and serves as a good example. If you have not done so, be sure to become familiar with the previous examples first.

PyTorch defines the layer normalization operation for an input matrix $X$, with shape batch size $(B)$ by hidden size $(H)$, as:

$$
y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta
$$

Where the mean $\mathrm{E}[x]$ and variance $\operatorname{Var}[x]$ are calculated for each sample in a batch, and $\gamma$ and $\beta$ are learnable vector weights with lengths equal to the hidden size. $\epsilon$ is a constant usually equal to $1 \mathrm{e}-05$.

As shown in a previous example, we can represent this using index notation:

$$
\begin{aligned}
m_{b} & =\frac{1}{H} \mathbf{1}_{h} x_{b h} \\
v_{b} & =\frac{1}{H} \mathbf{1}_{h}\left(x_{b h}-\mathbf{1}_{h} m_{b}\right)^{2} \\
y_{b h} & =\frac{x_{b h}-\mathbf{1}_{h} m_{b}}{\sqrt{v_{b}+\epsilon}} \gamma_{h}+\mathbf{1}_{b} \beta_{h}
\end{aligned}
$$

To make the problem more manageable, we are going to define additional intermediate tensor functions $\mu_{b h}$ and $\sigma_{b}$ :

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

![](https://cdn.mathpix.com/cropped/2024_05_07_c7c3c439079236dbfba7g-16.jpg?height=724&width=417&top_left_y=1107&top_left_x=190)

## Gradient of weights

Let's start with the easier gradients $\gamma$ and $\beta$ :

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

## The gradient of input $X$

Directly calculating the derivative of $y_{b h}$ with respect to $x_{p q}$ is quite complex and is an order-4 tensor. However, we don't need to construct this tensor fully since we can backpropagate the loss after each intermediate tensor function, simplifying the process. The backpropagated gradient is simpler because the loss is a scalar, meaning the gradient is, at most, an order-2 tensor.

To accomplish this, we'll start at the end of the dependency graph and calculate the Jacobian tensor at each intermediate stage, followed by calculating the backpropagated gradient. The goal is to obtain an expression of $\partial l / \partial x_{p q}$ in terms of $\partial l / \partial y_{p q}$.

Gradient of $\sigma$

The derivative of $y_{b h}$ with respect to $\sigma_{p}$ :

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

The derivative of $\sigma_{b}$ with respect to $v_{p}$ :

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

The function $\mu_{b h}$ is consumed by two functions, $v_{b}$ and $y_{b h}$, therefore we need to differentiate both functions by $\mu_{b h}$. First, the derivative of $v_{b}$ with respect to $\mu_{p q}$ :

$$
\begin{aligned}
\frac{\partial v_{b}}{\partial \mu_{p q}} & =\frac{\partial}{\partial \mu_{p q}}\left(\frac{1}{H} \mathbf{1}_{h} \mu_{b h}^{2}\right) \\
& =\frac{2}{H} \mathbf{1}_{h} \mu_{b h} \delta_{b p} \delta_{h q} \\
& =\frac{2}{H} \mu_{b q} \delta_{b p}
\end{aligned}
$$

Then derivative of $y_{b h}$ with respect to $\mu_{p q}$ :

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

The derivative with respect to $m_{p}$ :

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

In the first term, we have the sum $\mathbf{1}_{h} \mu_{p h}$ which can be shown to equal zero:

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

And finally, we move onto $x_{b h}$. Two functions, $m_{b}$ and $\mu_{b h}$, consume $x_{b h}$ and so we need to consider both. First, the derivative of $m_{b}$ with respect to $x_{p q}$ :

$$
\begin{aligned}
\frac{\partial m_{b}}{\partial x_{p q}} & =\frac{\partial}{\partial x_{p q}}\left(\frac{1}{H} \mathbf{1}_{h} x_{b h}\right) \\
& =\frac{\mathbf{1}_{q}}{H} \delta_{b p}
\end{aligned}
$$

And the derivative of $\mu_{b h}$ with respect to $x_{p q}$ :

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

## Summary

This article explores using tensor calculus to derive gradients in deep learning. It provides the tools to derive gradients of tensors of arbitrary order, including vectors and matrixes. We focus on Cartesian tensors for simplicity, which are sufficient for deep learning. We show how some tensor gradients can be equivalent to vectors and matrices, although these are not always sufficient to represent all expressions. We outline a procedure for calculating backpropagated gradients and apply these concepts through practical examples. The piece serves as both a reference and a practical guide.

## References

- The Matrix Calculus You Need For Deep Learning by Terence Parr and Jeremy Howard
- Matrix Differential Calculus with Applications in Statistics and Econometrics Book by Heinz Neudecker and Jan R. Magnus
- Tensor Calculus by David Kay
- Mathematical Methods for Physics and Engineering: A Comprehensive Guide by K. F. Riley, M. P. Hobson and S. J. Bence


## Appendix

## Functions, variables and values

It is common in the literature to use the same label for both a function and variable, e.g. $y=y(x)$ where $y(\quad)$ is the function and $y$ is the output variable. It is also common to not distinguish between variables and values. However, this ambiguity can fail sometimes - we will explore why.

Let's say we have two functions, $f_{1}\left(x_{1}\right)$ and $f_{2}\left(x_{2}\right)$. If we are told the inputs and outputs are always equal in value, does this mean $x_{1}=x_{2}$ and $f_{1}=f_{2} ?$

To demonstrate why it's a no, let's consider the derivatives. As stated, $f_{1}$ is a function of $x_{1}$ and independent of $x_{2}$. Summarily, $f_{2}$ a function of $x_{2}$ and independent of $x_{1}$. This means $\partial f_{1} / \partial x_{2}=0$ and $\partial f_{2} / \partial x_{1}=0$.

However, if we substituted based on value using $x_{1}=x_{2}, f_{1}=f_{2}$, then $\partial f_{1} / \partial x_{2}=\partial f_{1} / \partial x_{1}$ and as $f_{1}$ is a function of $x_{1}$, $\partial f_{1} / \partial x_{1}$ is not always zero. Therefore, sometimes it's important to keep the distinction between functions, variables and values.

## Commutative and associative properties of index notation

The following usual commutative and associative properties for addition and multiplication apply to index notation:

$$
\begin{aligned}
a_{i}+b_{i} & =b_{i}+a_{i} \\
\left(a_{i}+b_{i}\right)+c_{i} & =a_{i}+\left(b_{i}+c_{i}\right) \\
a_{i} b_{j} & =b_{j} a_{i} \\
a_{i} b_{i} & =b_{i} a_{i} \\
\left(a_{i} b_{j}\right) c_{k} & =a_{i}\left(b_{j} c_{k}\right) \\
\left(a_{i} b_{j}\right) c_{j} & =a_{i}\left(b_{j} c_{j}\right)
\end{aligned}
$$

