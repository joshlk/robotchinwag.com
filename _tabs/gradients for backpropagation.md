---
title: Gradients for Backpropagation Series
layout: post
icon: fas fa-star
order: 1
---

Here is a collection of posts explaining how to derive the gradients for backpropagation for function in deep learning.

Gradients and auto-differentiation are the backbone of how deep learning models "learn" using the backpropagation algorithm. However, good resources that explain the theory for functions that work with tensors are scarce, and tensors are used by almost all functions in deep learning.

Included in the blog series:

* [The Tensor Calculus You Need for Deep Learning]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}) explains how to derive gradients using tensor calculus.
* [A brief tour of backpropagation and multi-variable calculus]({% link _posts/2024-05-02-backpropagation-and-multivariable-calculus.md %}) provides a refresher to the background to the rest of the articles.
* [The Gradients of Layer Normalization]({% link _posts/2024-05-04-layer-normalization-deriving-the-gradient-for-the-backward-pass.md %}) is an good application of tensor calculus, as explained in the previous article on a more complex deep-learning function.

All examples of deriving gradients using tensor calculus:

* [Element-wise or pointwise functions]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#example-element-wise-functions)
* [Matrix multiplication]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#example-matrix-multiplication)
* [Inverse of matrix]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#example-matrix-inverse)
* [Cross-Entropy Loss]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#example-cross-entropy-loss)
* [Layer Normalization]({% link _posts/2024-05-04-layer-normalization-deriving-the-gradient-for-the-backward-pass.md %})

## Why is this Series Needed?

The gradient of a matrix with respect to another matrix is a tensor, and the corresponding chain rule involves tensor products that are not representable using matrices. Thus, a solid understanding of tensor calculus is necessary to get a complete picture of gradients in deep learning.

Many texts on deep learning rely on vector calculus to derive gradients, which involves differentiating vectors with respect to vectors and organising the resulting derivatives into matrices. However, this falls short of a systematic method to differentiate matrices or tensors and can confuse beginners. While matrix calculus, particularly [Magnus & Neudecker framework](https://www.google.com/search?client=firefox-b-d&q=Matrix+Differential+Calculus+with+Applications+in+Statistics+and+Econometrics+Book+by+Heinz+Neudecker+and+Jan+R.+Magnus), does offer an alternative by flattening matrices into vectors, it doesn't effectively extend to tensors, and the algebra is bloated. Therefore, this series introduces tensor calculus using index notation as a more robust and generalised solution for deriving gradients within deep learning.

Tensor calculus is unfamiliar to most practitioners, and the available literature relevant to deep learning is scarce. Acquiring the necessary background means navigating various disciplines, often using inconsistent notation and conventions. This complexity can be a significant barrier for newcomers seeking a single source of truth.

This blog series addresses this gap by providing a consistent framework that links multivariable calculus, backpropagation and tensor calculus. [Part 1]({% link _posts/2024-05-02-backpropagation-and-multivariable-calculus.md %}) offer a concise overview of the necessary background in multivariable calculus and backpropagation. For a more in-depth description, refer to the excellent article [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/). In [Part 2]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}), tensors and tensor calculus are introduced, aiming to provide a complete reference with [examples]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}#example-element-wise-functions) including [layer normalization]({% link _posts/2024-05-04-layer-normalization-deriving-the-gradient-for-the-backward-pass.md %}).

## Next

Start the series with [A brief tour of backpropagation and multi-variable calculus]({% link _posts/2024-05-02-backpropagation-and-multivariable-calculus.md %}) or skip into the meat of the theory with [The Tensor Calculus You Need for Deep Learning]({% link _posts/2024-05-03-the-tensor-calculus-you-need-for-deep-learning.md %}).
