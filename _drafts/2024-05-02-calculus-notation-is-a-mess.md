---
title: "Part 2: Tensors and Tensor Calculus Using Index Notation"
description: >-
  Introduction on Tensor calculus and backpropagation
#author: Josh Levy-Kramer
date: 2024-05-02 12:01:00 +0000
categories: [AI, Tensor Calculus]
tags: [ai, deep learning, maths, tensor calculus, automatic differentiation]  # TAG names should always be lowercase
pin: true
math: true
---

# Part 2: Tensors and Tensor Calculus using Index Notation

This article forms part of a series (<>) on differentiating and calculating gradients in deep learning. Part 1 (<>) introduced backpropagation and multivariable calculus, which sets up some of the ideas used in the article. Here, we get to the meat of the theory, tensors and tensor calculus using index notation. I have modified the index notation explained here for deep learning and deviates from other literature. For example, I allow dummy indices to be used more than twice in an expression, as this frequently occurs in deep learning. Some examples of using tensor calculus are shown at the bottom of the page (...)

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

### Gradient of weights

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

### The gradient of input $X$

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

### Gradient of $v$

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

### Gradient of $\mu$

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

### Gradient of $m$

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

### Gradient of $x$

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

### Conclusion

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

- [The Matrix Calculus You Need For Deep Learning by Terence Parr and Jeremy Howard](https://explained.ai/matrix-calculus/)
- [Matrix Differential Calculus with Applications in Statistics and Econometrics Book by Heinz Neudecker and Jan R. Magnus](https://www.google.com/search?client=firefox-b-d&q=Matrix+Differential+Calculus+with+Applications+in+Statistics+and+Econometrics+Book+by+Heinz+Neudecker+and+Jan+R.+Magnus)
- [Tensor Calculus by David Kay](https://www.google.com/search?client=firefox-b-d&q=Tensor+Calculus+by+David+Kay)
- [Mathematical Methods for Physics and Engineering: A Comprehensive Guide by K. F. Riley, M. P. Hobson and S. J. Bence](https://www.google.com/search?client=firefox-b-d&q=Mathematical+Methods+for+Physics+and+Engineering%3A+A+Comprehensive+Guide+by+K.+F.+Riley%2C+M.+P.+Hobson+and+S.+J.+Bence)


## Appendix

### Functions, variables and values

It is common in the literature to use the same label for both a function and variable, e.g. $y=y(x)$ where $y(\quad)$ is the function and $y$ is the output variable. It is also common to not distinguish between variables and values. However, this ambiguity can fail sometimes - we will explore why.

Let's say we have two functions, $f_{1}\left(x_{1}\right)$ and $f_{2}\left(x_{2}\right)$. If we are told the inputs and outputs are always equal in value, does this mean $x_{1}=x_{2}$ and $f_{1}=f_{2} ?$

To demonstrate why it's a no, let's consider the derivatives. As stated, $f_{1}$ is a function of $x_{1}$ and independent of $x_{2}$. Summarily, $f_{2}$ a function of $x_{2}$ and independent of $x_{1}$. This means $\partial f_{1} / \partial x_{2}=0$ and $\partial f_{2} / \partial x_{1}=0$.

However, if we substituted based on value using $x_{1}=x_{2}, f_{1}=f_{2}$, then $\partial f_{1} / \partial x_{2}=\partial f_{1} / \partial x_{1}$ and as $f_{1}$ is a function of $x_{1}$, $\partial f_{1} / \partial x_{1}$ is not always zero. Therefore, sometimes it's important to keep the distinction between functions, variables and values.

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
