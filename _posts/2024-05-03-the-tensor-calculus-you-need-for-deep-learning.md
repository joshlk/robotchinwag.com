---
title: "The Tensor Calculus You Need for Deep Learning"
description: >-
  Deriving the gradient for the backward pass using tensor calculus and index notation
#author: Josh Levy-Kramer
date: 2024-05-03 12:01:00 +0000
categories: [AI, Tensor Calculus]
tags: [ai, deep learning, maths, tensor calculus, index notation, automatic differentiation]  # TAG names should always be lowercase
pin: false
math: true
---

* [Into: series intro and more examples](/posts/gradients-in-deep-learning/)
* [Part 1: a brief tour of backpropagation and multi-variable calculus](/posts/backpropagation-and-multivariable-calculus/)
* [Part 2: The Tensor Calculus You Need for Deep Learning](/posts/the-tensor-calculus-you-need-for-deep-learning/)
* [Part 3: the Gradients of Layer Normalization](/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/)

This article forms part of a [series](/posts/gradients-in-deep-learning/) on differentiating and calculating gradients in deep learning. [Part 1](/posts/backpropagation-and-multivariable-calculus/) introduced backpropagation and multivariable calculus, which sets up some of the ideas used in this article. Here, we get to the meat of the theory: tensors and tensor calculus using index notation.

Note that the index notation explained here has been modified so its more appropriate for deep learning and deviates from other sources. For example, I allow dummy indices to be used more than twice in an expression, as this frequently occurs in deep learning. [Some examples](#example-element-wise-functions) of using tensor calculus are shown in the last sections of this page.

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

The bases of a matrix can be constructed by applying the [tensor product](https://mathworld.wolfram.com/VectorSpaceTensorProduct.html) $(\otimes)$ of two basis vectors. Don't worry about the details of how the tensor product works - we won't be using it much:

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

As you can see, the notation gets unwieldy very quickly when the number of dimensions grows larger than 2. We also need a way of defining operations between tensors. We will, therefore, turn to index notation to provide a more convenient way to represent tensors of any order.

## Index notation

Index notation provides a convenient algebra to work with individual elements or components of a tensor. It can also be used to easily define opperations between tensors, for example, a matrix multiplication can be expressed as:

$$
c_{i k}=a_{i j} b_{j k}
$$

Below, we explain the rules of index notation and how to understand the above expression.

Index notation (not to be confused with multi-index notation) is a simplified version of Einstein notation or Ricci calculus that works with Cartesian tensors. Importantly, not all the normal algebra rules apply with index notation, so we must take care when first using it.

Be aware that different authors use various flavours of index notation, so when reading other texts, look at the details of how they use the notation. The notation presented here derivates from others as its been addapted for Cartesian tensors and deep learning.

### Index notation rulebook

#### Tensor indices

- A tensor is written using uppercase curly font $\mathcal{T}$. Its components are written in lowercase, with its indices in subscript $t_{i j k}$. You can obtain the components of a tensor using square brackets e.g. $$ \left [ \mathcal{T}  \right ]_{i j k}=t_{i j k} $$.

- The number of indices represents the order of the tensor. For example $t_{i j k}$ has three indices $i, j$ and $k$, and so is an order- 3 tensor.
- The indices can use any letters, so without further context $t_{i j k}$ and $t_{a b c}$ refer to the same tensor and components.
- The range of the indices is either determined by context or can be kept arbitrary.
- Indices are labelled using lowercase Latin symbols.


#### Free indices

- Free indices describe a system of equations and imply that the expression is repeated, enumerating the index range. For example, if the index $i$ has the range $i=\{1,2,3\}$, then:

$$
a_{i}=b_{i} c_{i} \Leftrightarrow\left\{\begin{array}{l}
a_{1}=b_{1} c_{1} \\
a_{2}=b_{2} c_{2} \\
a_{3}=b_{3} c_{3}
\end{array}\right.
$$

- A free index must appear in every additive term in the equation. Most of the time you can determine the free indices by checking an equation's left-hand side (LHS) or the subject of an equation.
- A free index can be renamed if it is renamed in all additive terms.
- An equation can have multiple free indices, representing the system of equations that enumerate all indices.


#### Dummy indices

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
a_{i j} b_{i j}=\alpha_{i j} \beta_{i j} \quad\text {(no sum i) }
$$

#### Kronecker delta

- The Kronecker delta is an order-2 tensor whose components are 1 when the indices equal and zero otherwise:

$$
\delta_{i j}= \begin{cases}1 & \text { if } i=j \\ 0 & \text { otherwise }\end{cases}
$$

- The Kronecker delta is analogous to the identity matrix in matrix notation.
- The tensor is symmetric $\delta_{i j}=\delta_{j i}$.
- The Kronecker delta can be used to select a specific term in a summation. For example, we can select the $k$'th term of tensor $a$ in index $i$ using a contraction:

$$
\delta_{k i} a_{i j}=\delta_{k 0} a_{0 j}+\delta_{k 1} a_{1 j}+\ldots=\delta_{k k} a_{k j}=a_{k j}
$$

- More generally, if index $i$ is a dummy index and $k$ is a dummy or free index, using the Kronecker delta $\delta_{i k}$, we can replace index $i$ with all instances of index $k$. Consider these examples:

  - Replace $i$ with $k$ for multiple tensors:

  $$
  \delta_{k i} b_{i j} c_{i j}=b_{k j} c_{k j}
  $$

  

  * Replace $i$ with $j$ **or** $j$ with $i$:

  

$$
\delta_{i j} b_{i p} c_{q j}=b_{j p} c_{q j}=b_{i p} c_{q i}
$$

#### One-tensor

- The one-tensor is a tensor with the value 1 for all its components. It can be of any arbitrary order or shape:

$$
\mathbf{1}_{i \ldots j}=1
$$

- The one-tensor can be used to contract components of another tensor, for example, summing all components of $b$:

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

- The broadcast of an order-$M$ one-tensor with another order-$N$ one-tensor is equal to an order-$(M+N)$ one-tensor:

$$
a_{i j}=\mathbf{1}_{i} \mathbf{1}_{j}=\mathbf{1}_{i j}
$$

- A one-tensor can be dropped if a term already has a tensor with the same free or dummy index as the one-tensor, as this results in multiplying the terms by 1 :

$$
a_{i}=\mathbf{1}_{i j} b_{i j} c_{j}=b_{i j} c_{j}
$$

#### Substitution

When substituting one equation into another...

- You need to use the same free indices. For example, given $a_{i}=b_{i} c_{i}$ and $c_{j}=d_{j j}$, it would be wrong to conclude then $a_{i}=b_{i} d_{j j}$. Instead, we must change the free indices so they match to get $a_{i}=b_{i} d_{i i}$.
- You need to use different dummy indices. For example, given $$ a_{i}=b_{i j} \mathbf{1}_{j} c_{i} $$ and $$ c_{i}=d_{i j} e_{j} $$, we need to change $j$ to a new index $k$ in one of the equations to get $$ a_{i}=b_{i j} \mathbf{1}_{j} d_{i k} e_{k} $$ , otherwise we get $$ a_{i}=b_{i j} \mathbf{1}_{j} d_{i j} e_{j} $$ which is wrong.
- You cannot substitute into a contraction. For example, given $$ a=b_{k} c_{k} d_{k} $$ and $$ \beta=c_{k} d_{k}$$ it would be wrong to conclude $$ a=b_{k} \beta $$ because $$ b_{k} $$ is also part of the contraction.


#### Addition and multiplication

- Only tensors of the same order can be added together, for example, $a_{i j}+b_{i j}$ is valid while $a_{i j}+b_{i}$ is not. Except for adding a tensor with a scalar $a_{i j}+b$, which is implicitly broadcasted to the appropriate order $a_{i j}+b \mathbf{1}_{i j}$.
- The usual commutative and associative properties of addition and multiplications remain true ([see appendix](/posts/the-tensor-calculus-you-need-for-deep-learning/#commutative-and-associative-properties-of-index-notation)).
- Tensors don't distribute or factorise in the usual way. An expression can only be factorised, provided that all additive terms result in having the same tensor-order. For example:

$$
\begin{aligned}
a_{i} & =b_{i j} c_{i j}+b_{i j} d_{j} \\
& \neq b_{i j}\left(c_{i j}+d_{j}\right)
\end{aligned}
$$

But a one-tensor can be used to maintain the tensor-order of the terms:

$$
\begin{aligned}
a_{i} & =b_{i j} c_{i j}+b_{i j} d_{j} \\
& =b_{i j}\left(c_{i j}+\mathbf{1}_{i} d_{j}\right)
\end{aligned}
$$

#### Algebra of index notation

- Provided an equation, it is valid to apply a function to both sides:

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
f\left(a_{i}\right) & =f\left(b_{i k} c_{k}\right)
\end{aligned}
$$

- This includes adding a tensor to both sides, for example adding $\gamma_{i}$:

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
a_{i}+\gamma_{i} & =b_{i k} c_{k}+\gamma_{i}
\end{aligned}
$$

- Or multiplying both sides with a scalar $\lambda$:

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
a_{i} \lambda & =b_{i k} c_{k} \lambda
\end{aligned}
$$

- Or multiplying both sides using new free indices $\gamma_{j}$:

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
a_{i} \gamma_{j} & =b_{i k} c_{k} \gamma_{j}
\end{aligned}
$$

- Or multiply both sides with an existing free index to contract both sides $\gamma_{i}$:

$$
\begin{aligned}
a_{i} & =b_{i k} c_{k} \\
a_{i} \gamma_{i} & =b_{i k} c_{k} \gamma_{i}
\end{aligned}
$$

- It is not valid to contract an index that is already a dummy index, for example multiplying $\gamma_{k}$:

$$
\begin{gathered}
a_{i}=b_{i k} c_{k} \\
a_{i} \gamma_{k} \neq b_{i k} c_{k} \gamma_{k}
\end{gathered}
$$

To show this further, consider another example where $a_{k}=(2,-1)^{T} ; b_{k}=(0,1)^{T} ; \gamma=(0,1)^{T}$:

$$
\begin{aligned}
a_{k} \mathbf{1}_{k} & =b_{k} \mathbf{1}_{k} \\
a_{k} \mathbf{1}_{k} \gamma_{k} & \neq b_{k} \mathbf{1}_{k} \gamma_{k} \\
2 * 0-1 * 1 & \neq 0 * 0+1 * 1 \\
-1 & \neq 1
\end{aligned}
$$

- It is valid to use existing function identities, such as logarithmic or exponential identities e.g. $\log (a b)=\log (a)+\log (b)$ or e.g. $(a b)^{2}=a^{2} b^{2}$, but they should not change the components of a contraction. For example if $a_{i j}=b_{i} c_{j}$, then $a_{i j}^{2}=b_{i}^{2} c_{j}^{2}$ is valid; but if $a_{i j}=b_{i k} c_{j k}$, then $a_{i j}^{2}=b_{i k}^{2} c_{j k}^{2}$ is not valid.

#### Summary

We now have a way of representing order-N tensors using a convenient notation. It also allows us to define new operations between arbitrary ordered tensors, which was previously difficult to do using matrix notation without inventing new symbols.

Often, instead of referring to $a_{i j}$ as the components of an order-2 tensor, we will simply refer to $a_{i j}$ as the tensor. However, it should always be remembered that $a_{i j}$ are components and not the data structure itself.

### Matrix expressions in index notation

Here are some common matrix operations in index notation.

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

### Example: layer normalisation using index notation

PyTorch defines the layer normalization (layer norm) operation for an input matrix $X$, with shape batch size $B$ by hidden size $H$, as:

$$
y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta
$$

Where the mean $\mathrm{E}[x]$ and variance $\operatorname{Var}[x]$ are calculated for each sample in a batch, and $\gamma$ and $\beta$ are learnable vector weights with lengths equal to the hidden size. $\epsilon$ is a constant usually equal to $1 \mathrm{e}{-05}$.

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

Notice how we use the one-tensor for two different purposes here. The first is used to sum the terms like we used it for the mean. The second is used to broadcast the mean $m_{b}$ to an order-2 tensor $$ \mathbf{1}_{h} m_{b} $$, as the mean is invariant to the hidden dimension, and we want to subtract it from the order-2 tensor $x_{b h}$.

Now, we move to the definition of $\mathrm{y} . \gamma$ and $\beta$ are learnable vector weights with length equal to the hidden size, which means they are an order-1 tensor with length $H$. Using index notation, we can represent $y$ as:

$$
y_{b h}=\frac{x_{b h}-\mathbf{1}_{h} m_{b}}{\sqrt{v_{b}+\epsilon}} \gamma_{h}+\mathbf{1}_{b} \beta_{h}
$$

We use the one-tensor twice to broadcast $m_{b}$ and $\beta_{h}$ to an order-2 tensors.

## Tensor calculus

Tensor calculus allows us to consider how to differentiate a tensor with respect to a tensor.

Provided a function $\mathcal{Y}=\mathcal{F}(\mathcal{X})$, which takes as input an order-$M$ tensor $\mathcal{X}$ with components $x_{p \ldots q}$ and outputs a order-$N$ tensor $\mathcal{Y}$ with components $y_{i \ldots j}$; the derivative or Jacobian tensor is then of order-$(M+N)$:


$$
\frac{\partial \mathcal{Y}}{\partial \mathcal{X}}=\left[\frac{\partial \mathcal{Y}}{\partial \mathcal{X}}\right]_{i \ldots j p \ldots q} \hat{e}_{i \ldots j p \ldots q}=\frac{\partial y_{i \ldots j}}{\partial x_{p \ldots q}} \hat{e}_{i \ldots j p \ldots q}
$$

We are going to focus on index notation, so the derivative of a tensor with respect to a tensor is the quantity $\frac{\partial y_{i \ldots j}}{\partial x_{p \ldots q}}$ and $\frac{\partial}{\partial x_{p \ldots q}}$ is the tensor derivative operator. Below, we list the rules for applying the operator, and then we can get onto some examples.

### Tensor calculus rulebook

#### Tensor derivative operator

- The derivative operator should always use new free indices. For example, given the equation:

$$
y_{i . . j}=x_{i \ldots j}
$$

We pick the new free indices $a \ldots b$ when applying the derivative operator:

$$
\frac{\partial y_{i \ldots j}}{\partial x_{a \ldots b}}=\frac{\partial x_{i \ldots j}}{\partial x_{a \ldots b}}
$$

- The derivative of a tensor with respect to itself is a product of Krockener deltas. For example, for an order-1 tensor:

$$
\frac{\partial x_{i}}{\partial x_{p}}=\delta_{i p}
$$

This is because when $i \neq p, x_{i}$ is constant to $x_{p}$. In general, for the tensor of an order-$N$ tensor:

$$
\frac{\partial x_{i . . . j}}{\partial x_{p \ldots .}}=\underbrace{\delta_{i p} \ldots \delta_{j q}}_{N}
$$

#### Product and quotient rule

- We can use the product rule to obtain the tensor derivative of a product:

$$
\frac{\partial u_{a b} v_{c d}}{\partial x_{p q}}=\frac{\partial u_{a b}}{\partial x_{p q}} v_{c d}+u_{a b} \frac{\partial v_{c d}}{\partial x_{p q}}
$$

This can also be applied to a contraction ($b$ is a dummy index):

$$
\frac{\partial u_{a b} v_{b c}}{\partial x_{p q}}=\frac{\partial u_{a b}}{\partial x_{p q}} v_{b c}+u_{a b} \frac{\partial v_{b c}}{\partial x_{p q}}
$$

And the quotient rule for division:

$$
\frac{\partial}{\partial x_{p q}}\left(\frac{u_{a b}}{v_{c d}}\right)=\frac{1}{v_{c d}^{2}}\left(\frac{\partial u_{a b}}{\partial x_{p q}} v_{c d}-u_{a b} \frac{\partial v_{c d}}{\partial x_{p q}}\right)
$$

This also works for contractions ($u$ and $v$ can include dummy indices).

#### Chain rule

- We can also use the chain rule. For example, if a function outputs $y$ and takes $u$ as input and $u$ is a function of $x$ i.e. $y_{a \ldots b}\left(u_{c \ldots d}\left(x_{i \ldots j}\right)\right)$, then:

$$
\frac{\partial y_{a \ldots b}}{\partial x_{p \ldots q}}=\frac{\partial y_{a \ldots b}}{\partial u_{c \ldots d}} \frac{\partial u_{c \ldots d}}{\partial x_{p \ldots q}}
$$

Importantly, the two derivatives are contracted, i.e. $u_{c \ldots d}$ have the same indices in the first and second derivative terms. This mimics the summation shown in the multivariable calculus section, providing the weighted sum of all $x$ contributions to the change in $y$. Also note, we apply the derivative operator with new free indices $x_{p \ldots q}$.

- For the more general case of $n$ arbitrary nested functions $y_{a \ldots b}\left(u_{c \ldots d}\left(\ldots v_{e \ldots f}\left(x_{i \ldots j}\right)\right)\right)$, the chain rule is:

$$
\frac{\partial y_{a \ldots b}}{\partial x_{p \ldots q}}=\frac{\partial y_{a \ldots b}}{\partial u_{c \ldots d}} \frac{\partial u_{c \ldots d}}{\partial v_{e \ldots f}} \frac{\partial v_{e \ldots f}}{\partial x_{p \ldots q}}
$$

Importantly, the two contractions in the expression use different dummy indices and are separate contractions.

- If a function takes multiple tensors as input, then this must be considered when applying the chain rule. For example, provided with the function that takes two tensors $y_{a \ldots b}\left(u_{c \ldots d}\left(x_{i \ldots j}\right), v_{e \ldots f}\left(x_{i \ldots j}\right)\right)$, the chain rule would be:

$$
\frac{\partial y_{a \ldots b}}{\partial x_{p \ldots q}}=\frac{\partial y_{a \ldots b}}{\partial u_{c \ldots d}} \frac{\partial u_{c \ldots d}}{\partial x_{p \ldots q}}+\frac{\partial y_{a \ldots b}}{\partial v_{e \ldots f}} \frac{\partial v_{e \ldots f}}{\partial x_{p \ldots q}}
$$

### Summary

We are now at the point where we can combine tensor calculus with backpropagation! We have learned how to compute derivatives and apply the chain rule to tensors of any order.

In summary, to derive the backpropagated gradients of a tensor function, the steps are:

1. Translating the function into index notation
2. Calculating the derivative tensor of the function with respect to all its inputs
3. Assume the outputs are a dependency of a downstream scalar function $l$ and we are provided with the gradients of $l$ with respect to each of the outputs
4. Use the chain rule to determine the gradients of $l$ with respect to each of the inputs of the function.

Great! With this foundation, we're ready to move on to some practical examples.

## Example: element-wise functions

An element-wise or pointwise function applies an univariable function to all tensor components. For example, we could apply the trigonometric function sine to all tensor values. Let's consider an arbitrary pointwise function point:

$$
y_{i}=\operatorname{point}\left(x_{i}\right)
$$

Above $x_{i}$ and $y_{i}$ are order-1 tensors, but the following maths would apply to any order-$N$ tensor.

Firstly, to calculate the derivative of $y_{i}$ with respect to $x_{p}$, we differentiate using a new free index $p$:

$$
\begin{aligned}
\frac{\partial y_{i}}{\partial x_{p}} & =\frac{\partial \operatorname{point}\left(x_{i}\right)}{\partial x_{p}} \\
& =\frac{\partial \operatorname{point}\left(x_{i}\right)}{\partial x_{i}} \frac{\partial x_{i}}{\partial x_{p}} \\
& =\operatorname{point}^{\prime}\left(x_{i}\right) \delta_{i p}
\end{aligned}
$$

Where $\operatorname{point}^{\prime}$ is the derivative of the pointwise function, for example, if the pointwise function is sine, then its derivative is cos.

Secondly, we assume $y_{i}$ is a dependency of a downstream scalar function $l$ and that we are provided with the gradient $\partial l / \partial y_{i}$. We then use the chain rule to derive the backpropagated gradient of the input $\partial l / \partial x_{p}$:

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

### Gradient of $X$

Let's first obtain the derivative with respect to $X$, remeber to use new free indices for the derivative operator:

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

### Gradient of A

First, obtain the derivative with respect to $A$:

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

Notice that we can drop $$ \mathbf{1}_{n} $$ as $n$ is a free index, and $$ \mathbf{1}_{n} $$ equals 1 .

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

## Next

The [next part](/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/) of the series looks into applying these techniques to a more difficult function, layer normalisation.

## References

- [The Matrix Calculus You Need For Deep Learning by Terence Parr and Jeremy Howard](https://explained.ai/matrix-calculus/)
- [Matrix Differential Calculus with Applications in Statistics and Econometrics Book by Heinz Neudecker and Jan R. Magnus](https://www.google.com/search?client=firefox-b-d&q=Matrix+Differential+Calculus+with+Applications+in+Statistics+and+Econometrics+Book+by+Heinz+Neudecker+and+Jan+R.+Magnus)
- [Tensor Calculus by David Kay](https://www.google.com/search?client=firefox-b-d&q=Tensor+Calculus+by+David+Kay)
- [Mathematical Methods for Physics and Engineering: A Comprehensive Guide by K. F. Riley, M. P. Hobson and S. J. Bence](https://www.google.com/search?client=firefox-b-d&q=Mathematical+Methods+for+Physics+and+Engineering%3A+A+Comprehensive+Guide+by+K.+F.+Riley%2C+M.+P.+Hobson+and+S.+J.+Bence)


## Appendix

### Commutative and associative properties of index notation

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
