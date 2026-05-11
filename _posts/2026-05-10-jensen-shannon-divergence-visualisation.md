---
title: "Interactive Jensen–Shannon Divergence Visualisation"
description: >-
  An interactive visualisation of Jensen–Shannon divergence - the symmetric,
  always-finite cousin of KL. Shape two distributions and watch JSD, its
  ceiling of one bit, and the per-point contribution respond in real time.
#author: Josh Levy-Kramer
date: 2026-05-10 12:01:00 +0000
categories:
  - Statistics
tags:
  - jensen-shannon divergence
  - jensen-shannon
  - kl divergence
  - kullback-leibler
  - information theory
  - entropy
  - mutual information
  - probability
  - statistics
  - deep learning
  - maths
  - visualisation
  - interactive
math: true
---

This is a follow-up to the [interactive KL divergence explorer]({% post_url 2026-05-07-kl-divergence-visualisation %}). That one ended on two rough edges of Kullback–Leibler divergence: it's *asymmetric*, and it blows up to $\infty$ the moment one distribution puts mass where the other has none. Jensen–Shannon divergence is the standard fix for both. Drag the sliders and watch how it behaves.

<iframe src="{{ '/assets/jensen-shannon-explorer.html' | relative_url }}"
        width="100%"
        height="900"
        style="border: none;"
        loading="lazy"
        title="Interactive Jensen–Shannon divergence explorer: adjust two distributions and see JSD, its one-bit ceiling, and the pointwise integrand"></iframe>

For two probability distributions $P$ and $Q$, let $M = \tfrac12(P + Q)$ be their **mixture** - the distribution you get by flipping a fair coin and sampling from $P$ or $Q$ accordingly. The Jensen–Shannon divergence is

$$\mathrm{JSD}(P, Q) = \tfrac12 D_{\mathrm{KL}}(P \,\|\, M) + \tfrac12 D_{\mathrm{KL}}(Q \,\|\, M)$$

It's the average of two KL divergences, but instead of measuring $P$ against $Q$ directly, it measures each of them against the thing halfway between. That give you three properties KL doesn't have:

- **Symmetric.** Swapping $P$ and $Q$ leaves $M$ — and therefore $\mathrm{JSD}$ — unchanged.
- **Always finite, and bounded.** $M$ assigns zero probability only where *both* $P$ and $Q$ do, so neither KL term can ever divide by zero. In fact $0 \le \mathrm{JSD}(P, Q) \le \log 2$ — that's $\ln 2 \approx 0.693$ nats, or exactly **one bit**. Disjoint distributions equal one bit.
- $\boldsymbol{\sqrt{\mathrm{JSD}}}$ **is a true distance.** More on that below.

In the visualisation you control two skew-normal distributions, $P$ and $Q$, each with **Mean**, **Std**, **Skew** and **Truncate** sliders, plus a **Discretise** slider that bins both into histograms and computes the discrete JSD between the bin masses.

The upper plot draws the two densities. The lower plot draws the **integrand** $\tfrac12 p \log\frac{p}{m} + \tfrac12 q \log\frac{q}{m}$ — the pointwise contribution to JSD at each $x$. Unlike the KL integrand, this curve is non-negative everywhere and never infinite: the worst a single point can contribute is $\tfrac12 \max(p, q)\log 2$, which happens when one density is zero there and the other isn't.

Things worth playing with:

- **Make them disjoint.** Pull $P$ and $Q$ apart (or hit the *Nearly disjoint* preset). $\mathrm{JSD}$ climbs to $\ln 2$ — one bit — and stays there. In the [KL explorer]({% post_url 2026-05-07-kl-divergence-visualisation %}) the same move sent KL to $\infty$. Bounded versus undefined.
- **Truncate $Q$'s support.** Clip $Q$ to zero where $P$ still has mass. $D_{\mathrm{KL}}(P\|Q)$ would be $\infty$; $\mathrm{JSD}$ stays comfortably finite, because $M$ never vanishes there.
- **Coarsen the bins.** Turn on *Discretise* and widen the bins — $\mathrm{JSD}$ drops, because binning destroys information (the data-processing inequality). As bin width $\to 0$ it recovers the continuous value.

## Why it matters

Routing both distributions through their average $M$ is the whole trick: $M$ never assigns zero where $P$ or $Q$ has mass, so $D_{\mathrm{KL}}(P\|M)$ and $D_{\mathrm{KL}}(Q\|M)$ are both finite, and averaging the two makes the result symmetric. JSD is, roughly, "how badly does the *average* of $P$ and $Q$ describe each of them" — and that quantity has a clean interpretation.

**It's a mutual information.** Flip a fair coin $Z \in \{0, 1\}$; on $Z=0$ draw $X \sim P$, on $Z=1$ draw $X \sim Q$. Then

$$\mathrm{JSD}(P, Q) = I(X; Z)$$

— the information a single sample $X$ carries about *which distribution it came from*. If $P = Q$, a sample tells you nothing about the coin, and $\mathrm{JSD} = 0$. If $P$ and $Q$ have disjoint supports, a sample reveals the coin exactly — one bit — and $\mathrm{JSD} = \log 2$. JSD measures **distinguishability from a single draw**.

**Its square root is a metric.** $\sqrt{\mathrm{JSD}(P, Q)}$ is a distance metric that satisfies the triangle inequality: so you can actually do geometry with it — nearest-neighbour search over distributions, clustering documents by their word distributions, embedding distributions in a metric space.

**It answers a different question than KL.** $D_{\mathrm{KL}}(P\|Q)$ is the *expected log-likelihood-ratio under $P$*: "if the data genuinely comes from $P$, how wrong, on average, is someone who believes $Q$?" It's inherently directional and anchored to one reference distribution — which is exactly right when there *is* a ground truth and an approximation: maximum likelihood, cross-entropy loss, variational inference all minimise a KL of that shape. JSD asks a symmetric question instead — "how different are these two, with neither privileged?" — which is the right one when both sides are just empirical samples you want to compare: two text corpora, two model checkpoints, this month's traffic against last month's. And where KL answers "$\infty$ — incomparable" for distributions with different supports, JSD still gives a graded, bounded answer.

That bounded answer has a flip side worth knowing: once $P$ and $Q$ are fully disjoint, $\mathrm{JSD}$ is pinned at $\log 2$ and stops responding — its gradient vanishes — so pushing two disjoint distributions *towards* each other gives you nothing to descend.
