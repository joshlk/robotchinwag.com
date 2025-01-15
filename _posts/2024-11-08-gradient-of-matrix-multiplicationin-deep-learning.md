---
title: "Gradients of Matrix Multiplication in Deep Learning"
description: >-
  Deriving the gradients for the backward pass for matrix multiplication using tensor calculus
#author: Josh Levy-Kramer
date: 2024-11-08 12:01:00 +0000
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
pin: true
math: true
---

Matrix multiplication ([matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html)) is used all over the place in deep learning models, for example it's the basis of the [linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html). It is also common to use backpropagation to obtain the gradients of weights so we can update them using an optimiser. But how do we calculate those gradients for a simple matrix multiplication?

Lets say we have:


$$
Y = XA
$$


whereby:

* $X$ is the first matrix input and has a shape $I$ x $K$ 
* $A$ is the second matrix input (A.K.A the weights) and has a shape $K$ x $J$
* $Y$ is our matrix output with a shape $I$ x $J$

Lets now try to find the partial derivative of that expression:


$$
\frac{\partial Y}{\partial X} = \frac{\partial (XA)}{\partial X} = \mathord{?}
$$



But hang on, what does it mean to differentiate a matrix by a matrix? What type of object is the quantity $\partial Y / \partial X$. Is it a matrix, vector or something else? Also, we need some sort of product rule or chain rule to proceed, but does that exist for matrix functions?

![Thinking meme](/assets/img/thinking_meme.jpg){: width="300" }

Firstly, what does $\partial Y / \partial X$ mean? $Y$ is a function with a matrix output and $X$ is a matrix input to that function. The object $\partial Y / \partial X$ is the collection of gradients, and it has one gradient for each $Y$ component with respect to each $X$ component. So, the first gradient $\partial Y / \partial X$ is the derivative of the first component in $Y$ with respect to the first component in $X$ e.g. {::nomarkdown}$\partial [Y]_{0,0} / \partial [X]_{0,0}${:/}, using square brackets to obtain a matrix component. To enumerate all combinations of inputs and outputs $\partial Y / \partial X$ has  a shape of $I$ x $K$ x $K$ x $J$ elements.

But hold on. That means we need a 4-dimensional container to hold all of those gradients while matrices are 2-dimensional. This leads us to the realisation that $\partial Y / \partial X$ isn't a matrix 😱[^matrix_calculus], and we must use tensors! Tensors extend the idea of vectors and matrices to an arbitrary number of dimensions[^tensors].

![Enter the Tensor, meme](/assets/img/enter_the_tensor_meme.jpg){: width="300" }

Working mathematically with tensors can be daunting as we must let go of the nice matrix notation we learnt at school - it doesn't extend well to more than 2-dimensions. Instead, we can use index notation to describe tensors and tensor operations. I describe index notation in detail in a [previous article]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#index-notation), but briefly, instead of working with the containers (matrices and vectors), we work with the individual components. 

We can start with this alternative definition of matrix multiplication:


$$
[Y]_{i,j} = \sum_{k\in [1,K]} [X]_{i,k}[A]_{k,j}  \quad \text{for all} \; i \in [1,I], j \in [1,J]
$$


i.e. each element in the $Y$ matrix is calculated by summing the corresponding elements of $X$ multiplied by elements of $A$. The square brackets are the elements of a matrix, so $[Y]_{0,0}$ is the first element of the $Y$ matrix. The equation holds for all elements in $Y$ which are indexed using $i$ and $j$ which range between 1 to $I$, and 1 to $J$ respectively.

To simply the above, we can rewrite it using index notation:

$$
y_{i j} = x_{i k}a_{k j}
$$


We use lowercase symbols $y$, $x$ and $a$ which correspond to the elements of the matrices $Y$, $X$ and $A$. The subscripts indicate which element we are specifying, e.g. $y_{i j} = [Y]_{i, j} $. When a subscript is repeated in an expression on the right-hand side and not included on the left-hand side of the equation, it is implicitly summed over its entire range and called a dummy index - so we can drop the summation sign. We also drop the "for all" bit and use lowercase indices $i$ and $j$ which implicitly corresponds to the ranges $[i,I]$ and $[1,J]$ respectively.


Working with the elements of the matrices is easier as we can differentiate the expression using the tools we were taught at school. Let's go back to what we originally were interested in $\partial Y / \partial X$ - using index notation, we can express this as:


$$
\frac{\partial y_{i j}}{\partial x_{p q}}
$$

This quantity represents all the combinations of derivatives for each component of $Y$ and each component of $X$. We need to use new indices $p$ and $q$ for the $x$ in the denominator, but they still correspond to the ranges $[1,I]$ and $[1,K]$. The above represents the components of a 4-dimensional tensor as it has four indices $i$, $j$, $p$ and $q$.

Let's keep going with differentiating the expression:

$$
\frac{\partial y_{i j}}{\partial x_{p q}} = \frac{\partial (x_{i k}a_{k j})}{\partial x_{p q}}
$$


As $x$ and $a$ are scalars and $a$ is not a function of $x$ we can apply the [product rule]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#product-and-quotient-rule) to get:


$$
\frac{\partial (x_{i k}a_{k j})}{\partial x_{p q}} = a_{k j}\frac{\partial x_{i k}}{\partial x_{p q}}
$$


But how do we evaluate $\partial x_{i k}/ \partial x_{p q}$? Remember that $\partial x_{i k}/ \partial x_{p q}$ is an expression that enumerates all combinations of $x_{i k}$ with $x_{p q}$ i.e. all combinations of the elements in $X$ with itself. The derivative of a variable with itself is 1, while the derivative of an independent variable is 0. So $\partial x_{i k}/ \partial x_{p q}$ is 1 only when $i$ equals $p$ **and** $k$ equals $q$; otherwise, it's 0. To represent this, we use a special symbol called the [Kronecker Delta]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#kronecker-delta), $\delta_{i j}$ which is 1 when both its indices are equal and otherwise 0. So we have:


$$
\frac{\partial x_{i k}}{\partial x_{p q}} = \delta_{i p}\delta_{k q}
$$


and


$$
a_{k j}\frac{\partial x_{i k}}{\partial x_{p q}} = a_{k j}\delta_{i p}\delta_{k q}
$$

We can simplify further, which is easier to see if we bring back the summation sign:


$$
\begin{aligned}
a_{k j}\delta_{i p}\delta_{k q} &=  \sum_{k \in K} a_{k j}\delta_{i p}\delta_{k q} \\
 &= a_{0 j}\delta_{i p}\delta_{0 q} + a_{1 j}\delta_{i p}\delta_{1 q} + \dots \\
 &= a_{q j}\delta_{i p}
\end{aligned}
$$


As Kronecker Deltas are only 1 when the indices equal, in this case the second Kronecker Delta essentially picks a single value $a$ (see [this article for more details]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#kronecker-delta)). Which gives us:

$$
\frac{\partial y_{i j}}{\partial x_{p q}} = a_{k j}\frac{\partial x_{i k}}{\partial x_{p q}} = a_{q j}\delta_{i p}
$$


So, the derivative of $Y$ with respect to $X$ is equal to an element of $A$ only when $i$ and $p$ equal and otherwise zero. The above can't be expressed using matrix notation as the tensors are 4-dimensional, but once we use backpropagation (next section), the number of dimensions will reduce, and we will start to use matrices again.

I have glossed over many gory details, so I recommend reading my article on [The Tensor Calculus You Need For Deep Learning]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}).

## Backpropagation

![Backpropagation is all you need, meme](/assets/img/backprop_is_all_you_need_meme.jpg){: width="300" }

In deep learning, we often optimise our models using backpropagation. I have a previous article on [how backpropagation works]({% link _posts/2024-05-02-backpropagation-and-multivariable-calculus.md %}), but essentially, we assume a function consumes the output of the function we are interested in, which calculates a scalar loss $l$. We are provided with the gradient of the loss with respect to the output of our function, and we need to calculate the gradient of the loss with respect to all inputs of our function. So we are provided with the gradient, which is a 2-dimensional tensor or matrix:


$$
\frac{\partial l}{\partial y_{i j}}
$$


and we want to obtain the gradient of the loss with respect to $X$ and $A$ which are both matrices:


$$
\frac{\partial l}{\partial x_{p q}} \text{  ,  } \frac{\partial l}{\partial a_{q h}}
$$

To obtain those gradients we will need to use the chain rule, as $l$ is a function of $Y$ and $Y$ is a function of both $X$ and $A$. The tensor calculus chain rule is explained in detail [here]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#chain-rule) but, in short, if we are interested in the gradient of $t$ with respect to $s$ and there are $N$ intermediate variables $u_1, \dots, u_N$, then the gradient is calculated as follows:


$$
\frac{\partial t}{\partial s}=\sum_{n \in [1,N]}\frac{\partial t}{\partial u_{n}} \frac{\partial u_{n}}{\partial s}
$$


If we apply the rule to our situation above. We are interested in the gradient of the loss $l$ with respect to $x$ and we have all the components of $y$ as intermediate variables:

$$
\frac{\partial l}{\partial x_{p q}}=\sum_{i,j} \frac{\partial l}{\partial y_{i j}} \frac{\partial y_{i j}}{\partial x_{p q}} =\frac{\partial l}{\partial y_{i j}} \frac{\partial y_{i j}}{\partial x_{p q}}
$$

Notice how we sum over all components of $y$ as the indices $i$ and $j$, so we can use index notation and drop the summation sign. The equation holds for all combinations of $p$ and $q$ which range between $[1, I]$ and $[1,K]$ respectively.

Let now use everything we have learnt to finally derive the gradients of a matrix multiplication:

### The backpropagated gradient of $X$

To obtain the gradient of $X$, we can start with the result we obtained above:


$$
\frac{\partial y_{i j}}{\partial x_{p q}} = a_{q j}\delta_{i p}
$$


Next, we apply the [chain rule]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#chain-rule) so we can have an expression that relates $\partial l / \partial y_{i j}$ to $\partial l / \partial x_{p q}$:


$$
\frac{\partial l}{\partial x_{p q}} =\frac{\partial l}{\partial y_{i j}} \frac{\partial y_{i j}}{\partial x_{p q}}
$$


We can then substitute the previous expression and simplify it using the rules of index notation:


$$
\begin{aligned}
\frac{\partial l}{\partial x_{p q}} & =\frac{\partial l}{\partial y_{i j}} \frac{\partial y_{i j}}{\partial x_{p q}} \\
& =\frac{\partial l}{\partial y_{i j}} a_{q j}\delta_{i p} \\
& =\frac{\partial l}{\partial y_{p j}} a_{q j}
\end{aligned}
$$



So the matrix $\partial l / \partial x_{p q}$ equals some sort of product between the matrix $\partial l / \partial y_{p j}$ and the weights $A$. We now want to convert this to matrix notation. However, we are summing the second axis of both components, so this can't be represented as a matrix multiplication. In index notation, we can take the elements of a transposed matrix, $a_{q j} = (a^T)_{j q}$, so we can get:


$$
\frac{\partial l}{\partial x_{p q}} =\frac{\partial l}{\partial y_{p j}} (a^T)_{j q}
$$



Now the dummy indices are both on the inside of the expression like we first had above ($y_{i j} = x_{i k}a_{k j}$). So, we can then convert it to matrix notation like so (the square brackets indicate taking the elements of the matrix):


$$
\begin{aligned}
{\left[\frac{\partial l}{\partial X}\right]_{p q} } & =\left[\frac{\partial l}{\partial Y}\right]_{p j}\left[A^T\right]_{j q} \\
\frac{\partial l}{\partial X} & =\frac{\partial l}{\partial Y} A^T
\end{aligned}
$$



Therefore the gradient is calculated by multiplying the gradient of the loss $l$ with respect to the output $Y$ with the transpose of the weights $A$.

### The backpropagated gradient of $A$

We use the same procedure as above. First, obtain the derivative with respect to $A$:


$$
\begin{aligned}
\frac{\partial y_{i j}}{\partial a_{p q}} & =\frac{\partial (x_{i k} a_{k j})}{\partial a_{p q}} \\
& =x_{i k} \frac{\partial a_{k j}}{\partial a_{p q}} \\
& =x_{i k} \delta_{k p} \delta_{j q} \\
& =x_{i p} \delta_{j q}
\end{aligned}
$$



Then, we obtain the backpropagated gradient by assuming a downstream loss $l$ consumes the output $Y$:


$$
\begin{aligned}
\frac{\partial l}{\partial a_{p q}} & =\frac{\partial l}{\partial y_{i j}} \frac{\partial y_{i j}}{\partial a_{p q}} \\
& =\frac{\partial l}{\partial y_{i j}} x_{i p} \delta_{j q} \\
& =\frac{\partial l}{\partial y_{i q}} x_{i p}
\end{aligned}
$$



We now want to convert this to matrix notation. However, we are summing the first axis of both components, so this can't be represented as a matrix multiplication. We use the same trick we used in the previous section and take the transpose of $x_{i p}$ to swap the index order and then convert this to matrix notation:


$$
\begin{aligned}
{\left[\frac{\partial l}{\partial A}\right]_{p q} } & =\left[X^{T}\right]_{p i}\left[\frac{\partial l}{\partial Y}\right]_{i q} \\
\frac{\partial l}{\partial A} & =X^{T} \frac{\partial l}{\partial Y}
\end{aligned}
$$



Therefore, the gradient is calculated by taking the transpose of $X$ matrix multiplied by the gradient of the loss $l$ with respect to the output $Y$.

## Summary

Given a matrix multiplication used in a deep learning model:


$$
Y = XA
$$


The gradients used for backpropagation are:


$$
\frac{\partial l}{\partial X} =\frac{\partial l}{\partial Y} A^T
$$


and 


$$
\frac{\partial l}{\partial A} =X^{T} \frac{\partial l}{\partial Y}
$$



## Comparison with PyTorch

But are we right? Let's numerically compare the above equations with PyTorch. First, let's do a matmul in PyTorch:

```python
import torch

I, K, J = 128, 256, 512

# Create the tensors
torch.manual_seed(42)
X = torch.rand((I, K), requires_grad=True)
A = torch.rand((K, J), requires_grad=True)

# Do a matrix multiplication
Y = X @ A
```

To start backpropagation, we first need a function that calculates the loss. Here, we will sum Y for simplicity:

```python
l = Y.sum()
```

To obtain the backpropagated gradients, we can use PyTorch's `backward` method and then introspect the `.grad` property of a tensor.

```python
l.backward(inputs=(X, A, Y))
X.grad # This is X's backpropagated grad from l
A.grad # This is A's backpropagated grad from l
Y.grad # This is Y's backpropagated grad from l
```

Let's compare those grads with our derived equations above. So:

```
dldX = Y.grad @ A.T
dldA = X.T @ Y.grad
```

And we can check that they are indeed the same:

```python
torch.testing.assert_close(dldX, X.grad)
torch.testing.assert_close(dldA, A.grad)
```

## Next

If you would like to read more about calculating gradients using tensor calculus and index notation, please have a look at the [series introduction]({% link _tabs/gradients for backpropagation.md %}) or [The Tensor Calculus You Need for Deep Learning]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}).

## Footnotes

[^matrix_calculus]: Although [Magnus & Neudecker](https://www.google.com/search?client=firefox-b-d&q=Matrix+Differential+Calculus+with+Applications+in+Statistics+and+Econometrics+Book+by+Heinz+Neudecker+and+Jan+R.+Magnus) have created a framework to express the derivative of a matrix with respect to another matrix, the notation is clunky and doesn't extend to higher dimensional objects such as tensors. 

[^tensors]: The tensors we describe here are specifically N-dimensional [Cartesian tensors](https://en.wikipedia.org/wiki/Cartesian_tensor), which are simpler than tensors typically discussed in physics or mathematics. Focusing on Cartesian tensors removes the need to make a distinction between [covariant and contravariant indices,](https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors) and the same [transformation laws](https://phys.libretexts.org/Bookshelves/Relativity/General_Relativity_(Crowell)/04%3A_Tensors/4.04%3A_The_Tensor_Transformation_Laws) do not need to apply.
