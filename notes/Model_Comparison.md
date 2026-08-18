# Model Comparison (LOO / elpd)

### What LOO actually estimates
`elpd` is the expected log pointwise predictive density $\sum_i \log p(y_i \mid y_{-i})$, i.e. how well the model predicts each observation when that observation was held out. PSIS-LOO approximates all $n$ leave-one-out refits from a **single** fit by reweighting the posterior draws, which is why it needs a pointwise log-likelihood of shape `(chain, draw, n_obs)` rather than just the total.
- `p_loo` $\rightarrow$ effective number of parameters, from the gap between in-sample and LOO elpd. Should land near the real parameter count; wildly inflated means the fit is broken, not that the model is flexible.
- Pareto $k$ $\rightarrow$ per-observation reliability of the importance-sampling step. Above ~0.7 the reweighting has failed for that point and the estimate is not trustworthy.
- The absolute value of `elpd_loo` means nothing. Only differences between models scored on the same observations do.

### `pm.Potential` produces no log-likelihood at all
`pm.compute_log_likelihood` iterates over the model's **observed RVs**. A `pm.Potential` is not one, so for the mixture cure models it silently returns nothing — no group, no error, no warning. Hence the cure models record their pointwise likelihood as a `pm.Deterministic` alongside the Potential, which then gets lifted into a `log_likelihood` group after sampling.

The trap: the Deterministic and the Potential must be **separate nodes built from the same expression**. Wrapping the Deterministic variable makes the nutpie backend fail with `KeyError: 'obs'`.
```python
loglik = pt.switch(pt.eq(event, 1.0), log_lik_event, log_lik_censored)
pm.Deterministic("obs_loglik", loglik)
pm.Potential("obs", loglik)      # NOT pm.Potential("obs", <the Deterministic>)
```

### `idata_kwargs={"log_likelihood": True}` is ignored by external samplers
Honoured by `nuts_sampler="pymc"`, silently dropped on the nutpie / numpyro path — which is the default here. Passing the flag and assuming it worked gives an idata with no `log_likelihood` group and no indication why. The fix is to call `pm.compute_log_likelihood` explicitly after sampling, which works regardless of sampler.

### Aggregation level has to match what you are comparing
The piecewise Cox likelihood is a Poisson over **expanded subject-interval rows**, so LOO would score $N_{\text{long}}$ pseudo-observations instead of $n$ subjects — a different number of terms in the sum, making the elpd incomparable with any other model. Intervals are conditionally independent given the parameters, so summing a subject's rows recovers that subject's log-likelihood.

### Changing variables changes the density
The log-logistic AFT is fitted as $\log T \sim \text{Logistic}$, so its raw likelihood is $\log f_{\log T}$, not $\log f_T$. The Jacobian is a constant in the parameters, so dropping it is harmless *for sampling* — but it shifts the reported elpd onto a different measure:
$$\log f_T(t) = \log f_{\log T}(\log t) - \log t$$
with no correction for censored rows, since survival probabilities are invariant under a monotone transform. Uncorrected, this model's elpd was offset by $+\sum_{\text{uncensored}} \log t$ (~+325 on a 200-row test set) and won every comparison by a landslide.

### The general rule
LOO only compares densities computed **on the same measure, over the same observations, at the same aggregation level**. `az.compare` checks none of the three and will rank incomparable models without complaint — as it also will for a model that never sampled properly. Check divergences and Pareto $k$ before reading the ranking.
