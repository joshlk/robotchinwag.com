---
title: "Tensor calculus in deep-learning series"
description: >-
  Tensor calculus in deep-learning series
#author: Josh Levy-Kramer
date: 2024-05-01 12:01:00 +0000
categories: [AI, Tensor Calculus]
tags: [ai, deep learning, maths, tensor calculus, automatic differentiation]  # TAG names should always be lowercase
pin: true
Math: false
---

# Tensor calculus in deep-learning series

How do you calculate gradients in deep learning? Gradients and differentiation are fundamental to the theory of how deep learning models "learn" using an algorithim called back-propagation. Also, in deep learning, we use tensors, which are multi-dimensional data structures. Here, I present a series of articles that explain how to apply tensor calculus and back propagation to apply differentiation in deep learning and derive the "backwards" function used by frameworks like PyTorch.

The series is as follows:

* [Part 1: a brief tour of backpropagation and multi-variable calculus](/posts/part-1-backpropagation-and-multivariable-calculus/)
* [Part 2: Tensors and Tensor Calculus Using Index Notation](/posts/part-2-tensors-index-notation-and-tensor-calculus/)
* [Part 3: the Gradients of Layer Normalization](/posts/part-3-layer-normalization/)

Other examples of deriving gradients using tensor calculus and index notation:

* [Element-wise or pointwise functions](posts/part-2-tensors-index-notation-and-tensor-calculus/#example-element-wise-functions)
* [Matrix multiplication](/posts/part-2-tensors-index-notation-and-tensor-calculus/#example-matrix-multiplication)
* [Inverse of matrix](posts/part-2-tensors-index-notation-and-tensor-calculus/#example-matrix-inverse)
* [Cross-Entropy Loss](/posts/part-2-tensors-index-notation-and-tensor-calculus/#example-cross-entropy-loss)
* [Layer Normalization](/posts/part-3-layer-normalization/)



## Why this series is needed?

The gradient of a matrix with respect to another matrix is a tensor, and the corresponding chain rule involves tensor products that are not representable using matrices. Thus, a solid understanding of tensor calculus is necessary to get a complete picture of gradients in deep learning.

Many texts on deep learning rely on vector calculus to derive gradients, which involves differentiating vectors with respect to vectors and organizing the resulting derivatives into matrices. However, this falls short of a systematic method to differentiate matrices or tensors and can confuse beginners. While matrix calculus, particularly [Magnus \& Neudecker framework](https://www.google.com/search?client=firefox-b-d&q=Matrix+Differential+Calculus+with+Applications+in+Statistics+and+Econometrics+Book+by+Heinz+Neudecker+and+Jan+R.+Magnus), does offer an alternative by flattening matrices into vectors, it doesn't effectively extend to tensors, and the algebra is bloated. Therefore, this series introduces tensor calculus using index notation as a more robust and generalized solution for deriving gradients within deep learning.

Tensor calculus is unfamiliar to most practitioners, and the available literature relevant to deep learning is scarce. Acquiring the necessary background means navigating various disciplines, often using inconsistent notation and conventions. This complexity can be a significant barrier for newcomers seeking a single source of truth.

This article series addresses this gap by providing a consistent framework that links multivariable calculus, backpropagation and tensor calculus. [Part 1](/posts/part-1-backpropagation-and-multivariable-calculus/) offer a concise overview of the necessary background in multivariable calculus and backpropagation. For a more indepth description, refer to the excellent article [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/). In [Part 2](/posts/part-2-tensors-index-notation-and-tensor-calculus/), tensors and tensor calculus are introduced, aiming to provide a complete reference with [examples](posts/part-2-tensors-index-notation-and-tensor-calculus/#example-element-wise-functions) including [layer normalization](/posts/part-3-layer-normalization/).

## Next

Start the seriers with [Part 1: a brief tour of backpropagation and multi-variable calculus](/posts/part-1-backpropagation-and-multivariable-calculus/).
