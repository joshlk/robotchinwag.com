---
title: "The Tensor Calculus You Need for Deep Learning"
description: >-
  Deriving the gradient for the backward pass using tensor calculus and index notation
#author: Josh Levy-Kramer
date: 2024-05-03 12:01:00 +0000
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

When reading about deep learning theory, I was incredibly dissatisfied with the literature that explained how to derive gradients for backpropagation. It would mostly focus on vectors (and rarely matrices), but in deep-learning we use tensors everywhere! In part, this is probably since working with tensors mathematically isn't easy. This post is a spiritual successor to the excellent article [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) but for functions that work with tensors.

In this post, I introduce a complete and consistent framework for deriving gradients for backpropagation using tensor calculus. The [first section](#tensors) describes tensors (the type we use in deep learning) and introduces [index notation](#index-notation), a mathematical framework for working with them. Then we can get to [tensor calculus](#tensor-calculus) its self and show [some examples](#example-element-wise-functions).

This article forms part of a [series]({% link _tabs/gradients for backpropagation.md %}) on differentiating and calculating gradients in deep learning. [Part 1]({% link _posts/2024-05-02-backpropagation-and-multivariable-calculus.md %}) introduced backpropagation and multivariable calculus, which sets up some of the ideas used in this article. Here, we get to the meat of the theory: tensors and tensor calculus using index notation.

## Tensors

A tensor is a multi-dimensional ordered array of numbers, expanding the concept of a matrix into N-dimensions. Here and in deep learning, we are specifically talking about N-dimensional [Cartesian tensors](https://en.wikipedia.org/wiki/Cartesian_tensor), which are simpler than tensors typically discussed in physics or mathematics. Focusing on Cartesian tensors removes the need to make a distinction between [covariant and contravariant indices](https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors) and the same [transformation laws](https://phys.libretexts.org/Bookshelves/Relativity/General_Relativity_(Crowell)/04%3A_Tensors/4.04%3A_The_Tensor_Transformation_Laws) do not need to apply.

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

## Index Notation

Index notation provides a convenient algebra to work with individual elements or components of a tensor. It can also be used to easily define operations between tensors, for example, a matrix multiplication can be expressed as:

$$
c_{i k}=a_{i j} b_{j k}
$$

Below, I explain the rules of index notation and how to understand the above expression.

Index notation (not to be confused with multi-index notation) is a simplified version of [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) or [Ricci calculus](https://en.wikipedia.org/wiki/Ricci_calculus) that works with [Cartesian tensors](https://en.wikipedia.org/wiki/Cartesian_tensor).

> Not all the normal algebra rules apply with index notation, so we must take care when first using it.
{: .prompt-warning }

> There are many flavours of index notation with different subtleties. When reading other texts, look at the details of the rules they use. The notation presented here has been adapted for deep learning.
{: .prompt-warning }

### Index Notation Rulebook

#### Tensor Indices

- A tensor is written using uppercase curly font $\mathcal{T}$. Its components are written in lowercase, with its indices in subscript $t_{i j k}$. You can obtain the components of a tensor using square brackets e.g. $$ \left [ \mathcal{T}  \right ]_{i j k}=t_{i j k} $$.

- The number of indices represents the order of the tensor. For example $t_{i j k}$ has three indices $i, j$ and $k$, and so is an order- 3 tensor.
- The indices can use any letters, so without further context $t_{i j k}$ and $t_{a b c}$ refer to the same tensor and components.
- The range of the indices is either determined by context or can be kept arbitrary.
- Indices are labelled using lowercase Latin symbols.


#### Free Indices

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


#### Dummy Indices

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

> Typically, in Physics, you use Einstein notation to represent tensors, which prevents dummy indices from appearing more than twice and guarantees that the resulting tensor conforms to the [transformation laws](https://phys.libretexts.org/Bookshelves/Relativity/General_Relativity_(Crowell)/04%3A_Tensors/4.04%3A_The_Tensor_Transformation_Laws). We are using Cartesian tensors, which don't have this restriction, and in deep learning, it is common to sum over more than two tensors, so here,  we allow a dummy index to appear more than twice in an expression.
{: .prompt-warning }

#### Kronecker Delta

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

#### One-Tensor

- The one-tensor is a tensor with the value 1 for all its components. It can be of any order or shape:

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


#### Addition and Multiplication

- Only tensors of the same order can be added together, for example, $a_{i j}+b_{i j}$ is valid while $a_{i j}+b_{i}$ is not. Except for adding a tensor with a scalar $a_{i j}+b$, which is implicitly broadcasted to the appropriate order $a_{i j}+b \mathbf{1}_{i j}$.
- The usual commutative and associative properties of addition and multiplications remain true ([see appendix]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#commutative-and-associative-properties-of-index-notation)).
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

#### Algebra of Index Notation

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

### Matrix Expressions using Index Notation

Here are some common matrix operations expressed using index notation.

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

We often want to convert back and forth between the two notations and transposes are quite common. So a shorthand notation is often used: $a_{i j} = (a^T)_{j i}$ whereby $a^T$ are the components of the tensor $A^T$.

### Example: Layer Normalisation using Index Notation

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

## Tensor Calculus

Tensor calculus allows us to consider how to differentiate a tensor with respect to a tensor.

Provided a function $\mathcal{Y}=\mathcal{F}(\mathcal{X})$, which takes as input an order-$M$ tensor $\mathcal{X}$ with components $x_{p \ldots q}$ and outputs a order-$N$ tensor $\mathcal{Y}$ with components $y_{i \ldots j}$; the derivative or Jacobian tensor is then of order-$(M+N)$:


$$
\frac{\partial \mathcal{Y}}{\partial \mathcal{X}}=\left[\frac{\partial \mathcal{Y}}{\partial \mathcal{X}}\right]_{i \ldots j p \ldots q} \hat{e}_{i \ldots j p \ldots q}=\frac{\partial y_{i \ldots j}}{\partial x_{p \ldots q}} \hat{e}_{i \ldots j p \ldots q}
$$

We are going to focus on index notation, so the derivative of a tensor with respect to a tensor is the quantity $\frac{\partial y_{i \ldots j}}{\partial x_{p \ldots q}}$ and $\frac{\partial}{\partial x_{p \ldots q}}$ is the tensor derivative operator. Below, we list the rules for applying the operator, and then we can get onto some examples.

### Tensor Calculus Rulebook

#### Tensor Derivative Operator

- The derivative operator should always use new free indices. For example, given the equation:

$$
y_{i . . j}=x_{i \ldots j}
$$

We pick the new free indices $a \ldots b$ when applying the derivative operator:

$$
\frac{\partial y_{i \ldots j}}{\partial x_{a \ldots b}}=\frac{\partial x_{i \ldots j}}{\partial x_{a \ldots b}}
$$

- The derivative of a tensor with respect to itself is a product of Kronecker deltas. For example, for an order-1 tensor:

$$
\frac{\partial x_{i}}{\partial x_{p}}=\delta_{i p}
$$

This is because when $i \neq p, x_{i}$ is constant to $x_{p}$. In general, for the tensor of an order-$N$ tensor:

$$
\frac{\partial x_{i . . . j}}{\partial x_{p \ldots .}}=\underbrace{\delta_{i p} \ldots \delta_{j q}}_{N}
$$

#### Product and Quotient Rule

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

#### Chain Rule

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

## Example: Element-Wise Functions

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

## Example: Matrix Multiplication

Given $Y=X A$, we first need to convert it to index notation:

$$
y_{i j}=x_{i k} a_{k j}
$$

We have two gradients to derive as the function has two matrix inputs: $X$ and $A$.

### Gradient of $X$

Let's first obtain the derivative with respect to $X$, remember to use new free indices for the derivative operator:

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

### Gradient of $A$

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

## Next

Other examples of using tensor calculus to calculate gradients:

* [Linear layer]({% link _posts/2024-05-23-linear-layer-deriving-the-gradient-for-the-backward-pass.md %})
* [Layer Normalization]({% link _posts/2024-05-04-layer-normalization-deriving-the-gradient-for-the-backward-pass.md %})
* [Cross-Entropy Loss]({% link _posts/2024-11-22-crossentropy-loss-gradient.md %})
* [Inverse of matrix]({% link _posts/2024-11-22-maxtrix-inverse-gradient.md %})

## References

- [The Matrix Calculus You Need For Deep Learning by Terence Parr and Jeremy Howard](https://explained.ai/matrix-calculus/)
- [Matrix Differential Calculus with Applications in Statistics and Econometrics Book by Heinz Neudecker and Jan R. Magnus](https://www.google.com/search?client=firefox-b-d&q=Matrix+Differential+Calculus+with+Applications+in+Statistics+and+Econometrics+Book+by+Heinz+Neudecker+and+Jan+R.+Magnus)
- [Tensor Calculus by David Kay](https://www.google.com/search?client=firefox-b-d&q=Tensor+Calculus+by+David+Kay)
- [Mathematical Methods for Physics and Engineering: A Comprehensive Guide by K. F. Riley, M. P. Hobson and S. J. Bence](https://www.google.com/search?client=firefox-b-d&q=Mathematical+Methods+for+Physics+and+Engineering%3A+A+Comprehensive+Guide+by+K.+F.+Riley%2C+M.+P.+Hobson+and+S.+J.+Bence)


## Appendix

### Commutative and Associative Properties of Index Notation

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
