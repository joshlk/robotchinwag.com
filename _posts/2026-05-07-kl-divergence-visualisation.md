---
title: "KL Divergence Visualisation"
description: >-
  Visualisation Kullback–Leibler divergence of two input distributions
#author: Josh Levy-Kramer
date: 2026-05-07 12:01:00 +0000
categories:
  - Statistics
tags:
  - statistics
  - deep learning
  - maths
  - visualisation
pin: true
math: true
---

<iframe src="{{ '/assets/kl-divergence-explorer.html' | relative_url }}"
        width="100%"
        height="900"
        style="border: none;"
        loading="lazy"></iframe>

A visualisation of how KL divergence behaves when you change the input distributions.

For two probability distributions $P$ and $Q$ over the same space, the Kullback–Leibler divergence is

$$D_{\mathrm{KL}}(P \,\|\, Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$$

It measures the cost — in nats, if you use the natural log — of describing samples from $P$ using a code optimised for $Q$.

The lower plot shows the integrand $p(x)\log(p(x)/q(x))$ — the pointwise contribution. It can go negative where $Q$ exceeds $P$; only the signed area is guaranteed non-negative.

A few things worth playing with:

- **Asymmetry.** Narrow $Q$ with $P$ fixed and watch $D_{\mathrm{KL}}(P \,\|\, Q)$ explode, while $D_{\mathrm{KL}}(Q \,\|\, P)$ stays small. This is why the direction matters: minimising $D_{\mathrm{KL}}(Q \,\|\, P)$ is *mode-seeking* ($Q$ hides in one mode), minimising $D_{\mathrm{KL}}(P \,\|\, Q)$ is *mode-covering* ($Q$ must cover all of $P$).
- **Support.** Truncate $Q$ so it's exactly zero somewhere $P$ has mass, and $D_{\mathrm{KL}}(P \,\|\, Q) \to \infty$. This is the absolute-continuity condition that's usually glossed over — and why KL fails for disjoint distributions, motivating Jensen–Shannon and Wasserstein.
- **Discretisation.** Turn the bins on and KL drops, because binning destroys information (the data-processing inequality). As bin width $\to 0$, the discrete sum recovers the continuous integral.

Units are nats. Divide by $\ln 2$ for bits.
