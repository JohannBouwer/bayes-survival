# Evaluating Survival Models (held-out)

LOO answers "which model fits the data I have?". These answer "does it work on patients it never saw?" — a different question, and the two can disagree.

### C-index measures ordering, and nothing else
Harrell's C is the fraction of **comparable pairs** the model ranks correctly, ties counting as half. A pair $(i,j)$ is comparable only if you know who failed first: $T_i < T_j$ with $\delta_i = 1$. If $i$ was censored before $T_j$, the pair is dropped.
- It is invariant to any monotone transform of the risk score, so it **cannot see calibration**. A model predicting that everyone dies within three days scores 1.0 if the order is right.
- Not a proper scoring rule — nothing penalises reporting probabilities you do not believe.
- A covariate-free model (one KM curve for everyone) scores **exactly** 0.5, because every pair ties. Free check that the implementation is right.

### Harrell's C is not a pure property of the model
Censoring decides which pairs are comparable, so the same model on the same population scores differently under different follow-up. Only compare C-indices computed on the *same* test set. (Uno's C and IPCW variants exist to remove this dependence; unnecessary if every model is scored on one fold.)

### Brier scores the probability, so censoring has to be weighted back in
$\text{Brier}(t) = \text{mean}\big[(\hat S(t\mid x_i) - \mathbb{1}\{i \text{ alive at } t\})^2\big]$ — squared error on the predicted survival probability at one horizon. Proper, so unlike C-index it does penalise overconfidence.

The problem is anyone censored *before* $t$: you do not know their outcome. Dropping them biases the score, calling them survivors is simply wrong. IPCW reweights the ones you can classify, using $\hat G(u) = P(\text{not yet censored past } u)$ from a KM fit with censoring as the event:

| subject | contributes | weight |
|---|---|---|
| failed before $t$ | $(0-\hat S)^2$ | $1/\hat G(T_i)$ |
| known alive at $t$ | $(1-\hat S)^2$ | $1/\hat G(t)$ |
| censored before $t$ | nothing directly | mass carried by the two rows above |

### Brier has a base-rate floor — always benchmark against KM
The marginal KM curve already scores well whenever outcomes are lopsided, so absolute Brier values look deceptively good and 0 is not a reachable target. The only meaningful reference is the covariate-free row.

On SUPPORT2 (68% mortality), the log-normal AFT beat the KM baseline by:

| horizon | improvement |
|---|---|
| 30 d | **0.8%** |
| 180 d | 5.8% |
| 365 d | **8.1%** |

Same model, same covariates. Early on almost everyone is alive, so there is nothing for covariates to explain and the base rate does all the work. Quoting Brier without its baseline says almost nothing.

### Antolini's C-td differs from Harrell's only when curves cross
$C^{td} = P\big(\hat S_i(T_i) < \hat S_j(T_i) \mid T_i < T_j, \delta_i = 1\big)$ — each pair compared at the *earlier* subject's event time rather than through one scalar score. This is the metric the neural-survival benchmarks report (Kvamme et al. 2019), so it is the one to use when checking against published numbers.

It can only disagree with Harrell's C if predicted curves cross, and neither AFT nor PH can produce that:
- PH: $S(t\mid x) = S_0(t)^{\exp(X\beta)}$ — ordering fixed by the hazard ratio at every $t$.
- AFT: $S(t\mid x) = S_0(t/e^{X\beta})$ — covariates only stretch the time axis.

So for every model in this repo the two agree to ~4 decimals. Implementing C-td was still worth it: it turns "probably close enough" into a checked fact, and it is required the moment a crossing-curve model (Cox-Time, stratified, time-varying coefficients) enters the table.

### Calibration needs KM inside each bin
Bin the test set by predicted risk and compare mean predicted probability to observed. The observed side must come from a **KM fit within the bin**, not a raw proportion — censored subjects would otherwise be miscounted exactly as in the Brier case. Monotone and near-diagonal is the target; systematic offset means miscalibration that C-index is structurally blind to.

### The three answer different questions
`elpd` scores the whole predictive density in-sample, C-index scores only ranking out-of-sample, Brier scores absolute probability at one horizon. A flexible baseline hazard can win big on `elpd` while tying on C-index — better calibrated timing, same ability to say who fails first. Report all three, or say which one you chose and why.
