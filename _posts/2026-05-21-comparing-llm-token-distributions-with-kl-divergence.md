---
title: "Comparing LLM Token Distributions: An Interactive Zipf–KL Explorer"
description: >-
  An interactive visualisation of KL divergence between two Zipf-shaped token
  distributions — the kind a language model produces. Set each distribution's
  entropy, shuffle the rank ordering by a chosen Spearman correlation, and watch
  forward and reverse KL respond in real time.
#author: Josh Levy-Kramer
date: 2026-05-21 12:01:00 +0000
categories:
  - Statistics
tags:
  - kl divergence
  - kullback-leibler
  - zipf
  - zipfs law
  - language models
  - llm
  - logprobs
  - information theory
  - entropy
  - perplexity
  - rank correlation
  - probability
  - statistics
  - deep learning
  - maths
  - visualisation
  - interactive
math: true
---

This is a follow-up to the [interactive KL divergence explorer]({% post_url 2026-05-07-kl-divergence-visualisation %}), recast for the distribution an LLM (Large Language Model) produces: a probability over a vocabulary of tens of thousands of tokens. Drag the sliders and watch how KL between two such distributions behaves.

<iframe src="{{ '/assets/zipf-kl-explorer.html' | relative_url }}"
        width="100%"
        height="900"
        style="border: none;"
        loading="lazy"
        title="Interactive Zipf KL divergence explorer: two Zipf-distributed token distributions with adjustable entropy and rank correlation, showing forward and reverse KL and the per-rank contribution"></iframe>

At every step, a language model emits a vector of logits — one real number per vocabulary token — and a softmax turns them into a probability distribution. The log of those probabilities is what APIs hand back as logprobs. So a single forward pass gives you a full distribution $P$ over the vocabulary, not just the sampled token.

You often want to compare two such distributions, $P$ and $Q$, for example:

- the same prompt through two different models, or a model and its quantised / distilled / fine-tuned variant
- a policy against the model it was initialised from during [RLHF](https://arxiv.org/abs/2204.05862) (the KL penalty that stops it drifting)
- a small draft model against the target model in speculative decoding
- the same model at two temperatures

KL divergence is a way to score "how far has $Q$ drifted from $P$?". A language model's output probabilities can be approximated by a [Zipf distribution](https://en.wikipedia.org/wiki/Zipf%27s_law).

## Why Zipf?

Count how often each word appears in a large text corpus, sort by frequency, and the $i$-th most common word occurs with frequency roughly proportional to $1/i$. That is [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law) and commonly used when analysing frequency of tokens across a whole corpus. The general form assigns the token at rank $i$ a probability

$$p_i = \frac{i^{-s}}{Z}, \qquad Z = \sum_{k=1}^{N} k^{-s}$$

where $N$ is the vocabulary size, $s$ is the exponent (classic word-frequency Zipf has $s \approx 1$) and $Z$ is the normaliser.

Zipf is a decent approximation to an LLM's output probabilities, but the next-token distribution at a single position is only Zipf-*like* ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)).

In the visualisation both $P$ and $Q$ are Zipf distributions over the same $N$ tokens.

## Entropy sets the Zipf shape

Rather than expose the exponent $s$ directly, each distribution is controlled by its entropy:

$$H(P) = -\sum_i p_i \ln p_i$$

and the explorer solves for the $s$ that achieves it. Entropy is monotonic in $s$: at $s = 0$ the distribution is uniform with the maximum $H = \ln N$; as $s \to \infty$ it collapses onto the top token and $H \to 0$.

The **"="** button between the entropy sliders snaps $Q$'s entropy onto $P$'s.

## Rank correlation: when two models disagree on the ordering

Two real models rarely rank the vocabulary identically — they mostly agree on the obvious top tokens and diverge in the tail. The Rank corr. ρ control captures this: $Q$'s tokens are reordered relative to $P$'s until their [Spearman rank correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) reaches the value you set. For two rankings it is defined as:

$$\rho = 1 - \frac{6 \sum_i d_i^2}{N(N^2 - 1)}, \qquad d_i = \operatorname{rank}_P(i) - \operatorname{rank}_Q(i)$$

- $\rho = 1$ — identical ordering. Any KL comes purely from the entropy difference.
- $\rho \approx 0$ — independent orderings. KL is now dominated by *disagreement* about which token goes where.

The reshuffling jitters each token's log-rank, so the widely-spaced head stays put while the densely-packed tail scrambles first — mirroring how models agree on the likely tokens and disagree on the rest. The right pane plots $Q$'s probabilities in $P$'s rank order (with $P$ overlaid as a line): at $\rho = 1$ the bars trace $P$'s line; lower $\rho$ scatters them, and the per-rank plot lights up. (Note that with few elements a single permutation can't be perfectly decorrelated — at $N = 40$ the lowest reachable $\rho$ is around $0.3$, which the readout reports honestly.)

## How KL is calculated

For discrete distributions the KL divergence is a sum over the vocabulary:

$$D_{\mathrm{KL}}(P \,\|\, Q) = \sum_{i} p_i \, \ln \frac{p_i}{q_i}$$

Each token contributes $p_i \ln(p_i/q_i)$ — the bottom plot draws exactly these per-rank contributions.

Two properties carry over from the [continuous case]({% post_url 2026-05-07-kl-divergence-visualisation %}):

- **It's weighted by $P$**. Disagreements where $P$ is large dominate; disagreements out in the tail barely register. For token distributions this means KL is governed by the *head* — the handful of tokens the model actually considers.
- **It's asymmetric.** $D_{\mathrm{KL}}(P \,\|\, Q) \ne D_{\mathrm{KL}}(Q \,\|\, P)$ in general; the explorer shows both. Unlike the continuous explorer, KL here is **always finite** — both Zipf distributions have full support over the vocabulary, so no $q_i$ is ever zero.

## Things worth playing with

- **Same shape, different order.** Hit *Reordered only* (or press **=** then drop $\rho$). Both distributions have identical entropy, so every bit of KL is the models disagreeing about token ranks.
- **Same order, different sharpness.** *Temperature* keeps $\rho = 1$ and separates the entropies — KL with no reordering at all. This is the cost of running a model hotter or colder than its reference.
- **Watch the head dominate.** Increase $N$ and the per-rank plot stays concentrated near rank 1: KL is set by a few high-probability tokens, the long tail contributes almost nothing.
- **Forward vs reverse.** Make $Q$ much flatter than $P$ and compare the two KL boxes — the asymmetry is the same mode-seeking / mode-covering story from the [KL explorer]({% post_url 2026-05-07-kl-divergence-visualisation %}).

## Why it matters

When you fine-tune with an RLHF objective, the KL-to-reference penalty is precisely $D_{\mathrm{KL}}(\pi \,\|\, \pi_{\text{ref}})$ summed over the vocabulary at each step — keeping the policy's token distribution close to where it started ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)). Knowledge **distillation** trains the student by minimising a KL divergence to the teacher ([Gu et al., 2024](https://arxiv.org/abs/2306.08543)). **Speculative decoding** keeps outputs exact through a rejection step whose acceptance probability is one minus the total-variation distance between draft and target ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Chen et al., 2023](https://arxiv.org/abs/2302.01318)). And when you **quantise** a model, KL divergence from the full-precision original tracks how much the output distribution actually moved ([Dutta et al., 2024](https://arxiv.org/abs/2407.09141)).
