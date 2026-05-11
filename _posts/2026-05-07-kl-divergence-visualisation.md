---
title: "Interactive KL Divergence Visualisation"
description: >-
  An interactive visualisation of Kullback–Leibler divergence. Shape two
  distributions and watch forward vs reverse KL, the pointwise integrand,
  and the effects of asymmetry, support mismatch and discretisation.
#author: Josh Levy-Kramer
date: 2026-05-07 12:01:00 +0000
categories:
  - Statistics
tags:
  - kl divergence
  - kullback-leibler
  - information theory
  - entropy
  - probability
  - statistics
  - deep learning
  - maths
  - visualisation
  - interactive
pin: true
math: true
---

This page is an interactive explorer for building that intuition around Kullback–Leibler divergence: drag the sliders and watch the divergence respond in real time.

<iframe src="{{ '/assets/kl-divergence-explorer.html' | relative_url }}"
        width="100%"
        height="900"
        style="border: none;"
        loading="lazy"
        title="Interactive KL divergence explorer: adjust two distributions and see forward and reverse KL divergence and the pointwise integrand"></iframe>

An interactive explorer of how Kullback–Leibler (KL) divergence behaves when you change the input distributions.

For two probability distributions $P$ and $Q$, the Kullback–Leibler divergence (also called **relative entropy**) is

$$D_{\mathrm{KL}}(P \,\|\, Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$$

It measures how badly $Q$ approximates $P$, weighted by where $P$ actually puts its mass. Disagreements in regions $P$ considers likely count a lot; disagreements in regions $P$ considers unlikely barely count at all.

In the visualisation you control two skew-normal distributions, $P$ and $Q$, each with four sliders:

- **Mean** — shifts the distribution left/right.
- **Std** — widens or narrows it.
- **Skew** — pushes mass into one tail.
- **Truncate** — hard-clips the support to a window around the mean, so the density is exactly zero outside it.

There's also a **Discretise** slider that bins both distributions into histograms of a chosen width and computes the discrete KL between the bin masses instead of the continuous integral.

The upper plot draws the two densities. The lower plot draws the **integrand** $p(x)\log(p(x)/q(x))$ — the pointwise contribution to KL at each $x$.

Things worth playing with:

- **Asymmetry.** Narrow $Q$ with $P$ fixed and watch $D_{\mathrm{KL}}(P \,\|\, Q)$ explode, while $D_{\mathrm{KL}}(Q \,\|\, P)$ stays small. This is why the direction matters: minimising $D_{\mathrm{KL}}(Q \,\|\, P)$ is *mode-seeking* ($Q$ hides in one mode), minimising $D_{\mathrm{KL}}(P \,\|\, Q)$ is *mode-covering* ($Q$ must cover all of $P$).
- **Support.** Truncate $Q$ so it's exactly zero somewhere $P$ has mass, and $D_{\mathrm{KL}}(P \,\|\, Q) \to \infty$. This is the absolute-continuity condition that's usually glossed over — and why KL fails for disjoint distributions, motivating [Jensen–Shannon]({% post_url 2026-05-10-jensen-shannon-divergence-visualisation %}) and Wasserstein.
- **Discretisation.** Turn the bins on and KL drops, because binning destroys information (the data-processing inequality). As bin width $\to 0$, the discrete sum recovers the continuous integral.

If you've trained a classifier, you've minimised a KL divergence: **cross-entropy loss** is $D_{\mathrm{KL}}(P_{\text{data}} \,\|\, Q_{\text{model}})$ plus a constant entropy term.

For the symmetric, always-finite cousin of KL — the one that doesn't blow up on disjoint distributions — see the follow-up: [interactive Jensen–Shannon divergence visualisation]({% post_url 2026-05-10-jensen-shannon-divergence-visualisation %}).
