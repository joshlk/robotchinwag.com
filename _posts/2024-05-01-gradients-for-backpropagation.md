---
title: Gradients for Backpropagation Series
description: "Deriving the gradient for the backward pass using tensor calculus and index notation, series introduction"
date: 2024-05-01 12:01:00 +0000
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
Math: false
---

Here is a collection of posts explaining how to derive the gradients used in the backward function in deep learning.

Gradients and auto-differentiation are the backbone of how deep learning models "learn" using an algorithm called backpropagation. However, I haven't found good resources that explain the theory for functions that work with tensors, and tensors are used by almost all functions in deep learning.

Included in the blog series:

* [The Tensor Calculus You Need for Deep Learning](/posts/the-tensor-calculus-you-need-for-deep-learning/) explains how to derive gradients using tensor calculus.
* [A brief tour of backpropagation and multi-variable calculus](/posts/backpropagation-and-multivariable-calculus/) provides a refresher to the background theory needed.
* [The Gradients of Layer Normalization](/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/) is a nice application of using tensor calculus explained in the previous article on a more complex deep learning function.

All examples of deriving gradients using tensor calculus:

* [Element-wise or pointwise functions](/posts/the-tensor-calculus-you-need-for-deep-learning/#example-element-wise-functions)
* [Matrix multiplication](/posts/the-tensor-calculus-you-need-for-deep-learning/#example-matrix-multiplication)
* [Inverse of matrix](/posts/the-tensor-calculus-you-need-for-deep-learning/#example-matrix-inverse)
* [Cross-Entropy Loss](/posts/the-tensor-calculus-you-need-for-deep-learning/#example-cross-entropy-loss)
* [Layer Normalization](/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/)

## Why this series is needed?

The gradient of a matrix with respect to another matrix is a tensor, and the corresponding chain rule involves tensor products that are not representable using matrices. Thus, a solid understanding of tensor calculus is necessary to get a complete picture of gradients in deep learning.

Many texts on deep learning rely on vector calculus to derive gradients, which involves differentiating vectors with respect to vectors and organising the resulting derivatives into matrices. However, this falls short of a systematic method to differentiate matrices or tensors and can confuse beginners. While matrix calculus, particularly [Magnus \& Neudecker framework](https://www.google.com/search?client=firefox-b-d&q=Matrix+Differential+Calculus+with+Applications+in+Statistics+and+Econometrics+Book+by+Heinz+Neudecker+and+Jan+R.+Magnus), does offer an alternative by flattening matrices into vectors, it doesn't effectively extend to tensors, and the algebra is bloated. Therefore, this series introduces tensor calculus using index notation as a more robust and generalised solution for deriving gradients within deep learning.

Tensor calculus is unfamiliar to most practitioners, and the available literature relevant to deep learning is scarce. Acquiring the necessary background means navigating various disciplines, often using inconsistent notation and conventions. This complexity can be a significant barrier for newcomers seeking a single source of truth.

This blog series addresses this gap by providing a consistent framework that links multivariable calculus, backpropagation and tensor calculus. [Part 1](/posts/backpropagation-and-multivariable-calculus/) offer a concise overview of the necessary background in multivariable calculus and backpropagation. For a more in-depth description, refer to the excellent article [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/). In [Part 2](/posts/the-tensor-calculus-you-need-for-deep-learning/), tensors and tensor calculus are introduced, aiming to provide a complete reference with [examples](/posts/the-tensor-calculus-you-need-for-deep-learning/#example-element-wise-functions) including [layer normalization](/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/).

## Next

Start the series with A brief tour of backpropagation and multi-variable calculus,/posts/backpropagation-and-multivariable-calculus/) or skip into the meat of the theory with [A brief tour of backpropagation and multi-variable calculus](/posts/backpropagation-and-multivariable-calculus/).
