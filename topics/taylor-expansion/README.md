# Topic — Taylor Expansion

## Question

Starting from the Taylor remainder, study how one exact error object successively
acquires a representation, an asymptotic order, an upper bound, a propagation rule, and
an optimizable error model.

The core object is

\[
P_n(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k,
\qquad
R_n(x)=f(x)-P_n(x).
\]

Always record the function \(f\), expansion point \(a\), order \(n\), and evaluation
point \(x\). Every remainder formula also carries its own smoothness and interval
assumptions.

## Error map

- **Reference:** the exact mathematical value \(f(x)\).
- **Approximation:** the Taylor polynomial \(P_n(x)\), or its floating-point implementation \(\widehat P_n(x)\).
- **Primary error:** \(R_n(x)=f(x)-P_n(x)\).
- **Additional errors:** coefficient approximation, floating-point rounding, evaluation order, and cancellation.
- **Controls:** order \(n\), expansion point \(a\), step or evaluation distance, representation, and computing precision.
- **Optimization question:** at a given cost, how should these controls be chosen to minimize the total error?

## Study notes

Numbers are only this pass's reading order; all notes are complete.

| Note | Content |
| --- | --- |
| [00 — Error language](notes/00_error_language.md) | Distinguish actual error, representation, asymptotic order, big-\(O\), little-\(o\), and bound |
| [01 — Lagrange remainder](notes/01_lagrange_remainder.md) | Pin down the expansion object, interval, smoothness assumptions, and the unknown midpoint \(\xi\) |
| [02 — Integral remainder](notes/02_integral_remainder.md) | Compare what the integral form, the Lagrange form, and a worst-case bound each retain |
| [03 — Peano remainder](notes/03_peano_remainder.md) | Generalize from differentiability to the local asymptotic remainder \(R_n=o(h^n)\) |
| [04 — Bound tightness](notes/04_bound_tightness.md) | Compare actual error with a generic bound; check validity and tightness |
| [05 — Error propagation](notes/05_error_propagation.md) | Local linearization, computation chains, conditioning, stability, and correlated error |
| [06 — Control and optimization](notes/06_control_and_optimization.md) | Choose the optimal step against truncation, roundoff, cancellation, and random noise |

## First application: numerical differentiation

The forward difference

\[
D_hf(x)=\frac{f(x+h)-f(x)}{h}
\]

places Taylor truncation error and floating-point cancellation into one model:

\[
E(h)\approx C_1h+C_2\frac{u}{h}.
\]

The goal is to explain why an optimal step exists and how \(C_1,C_2\) relate to the
function scale and the specific computing model.

A central difference with correlated noise then unifies deterministic bias and random
variance as

\[
\operatorname{MSE}(h)
=
\left(\frac{\sinh h}{h}-1\right)^2
+
\frac{\sigma^2(1-\rho)}{2Nh^2}.
\]

This step verifies the bias–variance decomposition, the \(N^{-1/2}\) averaging law,
correlated-noise cancellation, and the statistically optimal step.

### Implementation ownership

When this experiment enters the coding stage it follows
`framework/implementation_learning_protocol.md`:

- the user first explains back the total error model and the experiment pseudocode;
- the agent scaffolds function signatures, test interfaces, data logging, and plotting;
- the user implements the Taylor / finite-difference core;
- before changing \(h\) or precision, the user predicts the error direction, magnitude, and curve shape;
- after review, the user rewrites one or two key evaluators closed-book, then compares by diff and tests.

## Topic exit criteria

On finishing the first pass of this topic, one should be able to:

1. state the Taylor remainder without conflating equality, asymptotics, and bound;
2. derive at least one remainder representation from assumptions;
3. judge whether an error bound is valid and whether it is tight;
4. write the truncation–roundoff error model for numerical differentiation;
5. explain why the optimal step exists, via both derivation and a reproducible experiment;
6. distinguish a single estimator's sample count \(N\) from the Monte Carlo repetition count \(M\) used to evaluate the estimator.

## Current status (2026-07-28)

- Notes 00–06 complete and archived;
- theory, implementation, pre-run predictions, Monte Carlo validation, and error attribution complete;
- both experiments retain raw CSV, metadata, and images;
- persistence tests and the closed-book rewrite complete;
- the first pass of this topic is formally closed;
- next entry: vector error propagation, the Jacobian, and Softmax.
