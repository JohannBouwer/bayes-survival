# bayes-survival

> [!NOTE]
> **Heads up:** This is a project I completed for my own development / learning. The models work and have been tested, but no claims are made about production-readiness, computational efficiency, or suitability for any particular use case.

Bayesian survival analysis models built on [PyMC](https://www.pymc.io/), with comparisons against [lifelines](https://lifelines.readthedocs.io/) frequentist equivalents. Inspired by [pymc-survival](https://github.com/pymc-labs/pymc-survival).

| Family | Models | |
|---|---|---|
| [Nonparametric](#nonparametric) | `KaplanMeierModel`, `NelsonAalenModel` | Conjugate — closed-form posterior, no MCMC |
| [AFT](#accelerated-failure-time-aft) | `Weibull`, `LogNormal`, `LogLogistic` + `AFTModel` | Covariates act on the time scale |
| [Hierarchical AFT](#hierarchical-aft) | the same three, as `Hierarchical…AFTModel` | Partial pooling over covariate groups |
| [Cox PH](#cox-proportional-hazards) | `PiecewiseCoxPHModel` | Piecewise-constant baseline, random-walk smoothing |
| [Mixture cure](#mixture-cure-models) | `LogNormal`, `Weibull`, `LogLogistic` + `CureModel` | Populations where some subjects never fail |

Every model shares one interface, ranks against the others with [`az.compare`](#model-comparison), and scores on held-out data through [`bayes_survival.metrics`](#held-out-evaluation). Worked examples for each are in [`notebooks/`](notebooks/README.md).

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JohannBouwer/bayes-survival.git
cd bayes-survival
uv sync
```

`uv sync` installs the core library — PyMC, PyTensor, NumPy, pandas, SciPy, ArviZ, lifelines — plus `nutpie[pymc]`, the default NUTS sampler. Optional extras:

| Command | Adds |
|---|---|
| `uv sync --extra notebook` | JupyterLab, matplotlib, statsmodels, scikit-learn — everything the notebooks need |
| `uv sync --extra samplers` | JAX and NumPyro, enabling `fit(nuts_sampler="numpyro")` |
| `uv sync --extra dev` | pytest, ruff |

> [!IMPORTANT]
> Each `uv sync --extra …` **replaces** the previous environment rather than adding to it. To combine extras, pass them in one command (`uv sync --extra notebook --extra dev`) or use `uv sync --all-extras`.

---

## Models

### Nonparametric

Bayesian nonparametric estimators that require no distributional assumptions. Both use conjugate priors, giving exact closed-form posteriors with no MCMC required.

#### `KaplanMeierModel`

Estimates the survival function via a Beta-Binomial conjugate model. At each distinct event time `t_j`, the conditional hazard `h_j = P(event at t_j | survived to t_j)` gets an independent Beta prior:

```
Prior:      h_j ~ Beta(α, β)
Posterior:  h_j | data ~ Beta(α + d_j, β + n_j - d_j)
S(t) = ∏_{t_j ≤ t} (1 - h_j)
```

| Prior key | Role | Default |
|-----------|------|---------|
| `h` | Beta prior on each conditional hazard | `Beta(1, 1)` — Uniform |

#### `NelsonAalenModel`

Estimates the cumulative hazard via a Gamma-Poisson conjugate model. At each event time `t_j`, the hazard increment `λ_j` gets an independent Gamma prior:

```
Prior:      λ_j ~ Gamma(α, β)
Posterior:  λ_j | data ~ Gamma(α + d_j, β + n_j)
H(t) = ∑_{t_j ≤ t} λ_j,   S(t) = exp(-H(t))
```

| Prior key | Role | Default |
|-----------|------|---------|
| `h` | Gamma prior on each hazard increment | `Gamma(0.1, 0.1)` — vague |

```python
from bayes_survival import KaplanMeierModel, NelsonAalenModel
import numpy as np

times = np.linspace(0, 36, 200)

km = KaplanMeierModel()
km.fit(t, event)
pred = km.predict_survival_function(times=times)
pred.mean, pred.hdi_lower, pred.hdi_upper       # each (1, n_times)

km.survival_probability(t=12.0)                       # single time point
km.posterior_mean_survival                            # (times, S) — closed form, no sampling
km.sample_posterior_survival(times, n_samples=2000)   # (n_samples, n_times)

na = NelsonAalenModel()
na.fit(t, event)
na.predict_cumulative_hazard(times=times)
na.predict_survival_function(times=times)       # via exp(-H(t))
na.posterior_mean_cumulative_hazard             # (times, H) — closed form
```

### Accelerated Failure Time (AFT)

All AFT models share the same interface: an intercept is added automatically, and positive `β_j` corresponds to longer expected survival times.

#### `WeibullAFTModel`

```
S(t | x) = exp( -(t / exp(Xβ))^α )
```

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `beta` | Log-scale coefficients (+ intercept) | `Normal(μ=0, σ=5)` |
| `alpha` | Weibull shape (α > 1: increasing hazard) | `Gamma(α=5, β=2)` |

#### `LogNormalAFTModel`

```
S(t | x) = Φ((Xβ - log(t)) / σ)
```

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `beta` | Log-mean coefficients (+ intercept) | `Normal(μ=0, σ=5)` |
| `sigma` | Spread of log-event times | `Gamma(α=5, β=2)` |

#### `LogLogisticAFTModel`

```
S(t | x) = 1 / (1 + (t / exp(Xβ))^α)
```

The hazard is non-monotonic (rises then falls), making this model suitable when event rates peak at some intermediate time. Fitted via the log-time reparameterization: `log(T) | x ~ Logistic(μ=Xβ, s=1/α)`.

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `beta` | Log-scale coefficients (+ intercept) | `Normal(μ=0, σ=5)` |
| `alpha` | Shape — controls both tail heaviness and hazard peak location | `Gamma(α=5, β=2)` |

```python
from bayes_survival import WeibullAFTModel
import numpy as np
import pymc as pm

# Inspect default priors before fitting
WeibullAFTModel.default_priors
# {'alpha': (Gamma, {'alpha': 5, 'beta': 2}), 'beta': (Normal, {'mu': 0, 'sigma': 5})}

model = WeibullAFTModel()                                        # defaults, or override any prior:
model = WeibullAFTModel(priors={"alpha": (pm.HalfNormal, {"sigma": 1})})
model.fit(X_train, t_train, event_train, draws=1000, tune=1000, chains=4)

# Survival function: mean + 94% HDI over a time grid
pred = model.predict_survival_function(X_test, times=np.linspace(0.1, 36, 200))
pred.mean, pred.hdi_lower, pred.hdi_upper       # each (n_obs, n_times)

model.survival_probability(X_test, t=12.0)                   # single time point
model.conditional_event_probability(X_test, t=6.0, T=24.0)   # P(event by T | alive at t)
model.sample_predicted_event_times(X_test)                   # (n_samples, n_obs)
```

These same calls work on every AFT, Cox, and cure model below.

### Hierarchical AFT

Hierarchical variants of each AFT model. Covariates belonging to a `HierarchySpec` group are drawn from `Normal(mu_group, sigma_group)`, where the hyper-parameters are themselves given priors — enabling **partial pooling** across the group. Sparse groups borrow strength from data-rich groups instead of being estimated in isolation.

The likelihood is unchanged from the corresponding flat model; only the priors on the grouped coefficients differ:

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `beta` (intercept + ungrouped) | Flat prior | `Normal(μ=0, σ=5)` |
| `mu_{group}` | Hyper-prior on group mean | `Normal(μ=0, σ=1)` |
| `sigma_{group}` | Hyper-prior on group std | `HalfNormal(σ=1)` |
| `alpha` / `sigma` | Shape, exactly as in the flat model | `Gamma(α=5, β=2)` |

`fit()` requires a `pd.DataFrame` so covariate names can be resolved. Prediction methods accept either a DataFrame or a plain NumPy array (columns in the same order as the training DataFrame).

```python
from bayes_survival import HierarchicalWeibullAFTModel, HierarchySpec

# Covariate groups that should share a hyper-prior
hierarchies = [
    HierarchySpec(name="product_type", covariate_names=["electronics", "clothing", "books"]),
    HierarchySpec(name="day_of_week", covariate_names=["mon", "tue", "wed", "thu"]),
]

model = HierarchicalWeibullAFTModel(hierarchies=hierarchies)
model.fit(df_train, t_train, event_train, draws=1000, tune=1000, chains=4)
model.predict_survival_function(df_test, times=np.linspace(0.1, 36, 200))
```

### Cox Proportional Hazards

#### `PiecewiseCoxPHModel`

A piecewise-constant Bayesian Cox PH model. The hazard is:

```
h(t | x) = h_0(t) · exp(Xβ)
```

where `h_0(t)` is piecewise constant over `K` intervals and `log h_0` follows a Gaussian Random Walk across intervals (smoothness prior). No intercept is added to `β`; the baseline hazard absorbs it.

Fitting uses the Poisson likelihood equivalence: data are expanded to long format (one row per observation-interval pair while at risk) and event counts are modelled as Poisson with rate `h_k · exp(Xβ) · exposure`.

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `log_baseline` | Log baseline hazard per interval (GRW) | `GaussianRandomWalk(sigma=grw_sigma)` |
| `grw_sigma` | Random-walk step-size (smoothness) | `HalfNormal(σ=1)` |
| `beta` | Log-hazard coefficients | `Normal(μ=0, σ=1)` |

```python
from bayes_survival import PiecewiseCoxPHModel

model = PiecewiseCoxPHModel(n_intervals=10)            # cut points at event-time quantiles
model = PiecewiseCoxPHModel(cuts=[6.0, 12.0, 24.0])    # or supply interior cut points explicitly
model.fit(X_train, t_train, event_train, draws=1000, tune=1000, chains=4)
```

Prediction uses the shared interface; `sample_predicted_event_times` draws via the piecewise-exponential inverse CDF.

### Mixture Cure Models

Some datasets contain individuals who will never experience the event — they are "cured". Standard AFT models cannot represent this: they assign non-zero hazard to all individuals at all future times, so the survival function eventually reaches zero for everyone.

A **mixture cure model** splits the population into two latent groups:

```
S_mix(t | x) = π(x) · S_u(t | x) + (1 - π(x))
```

where `π(x) = sigmoid(α + X·β_cure)` is the probability of being susceptible, `S_u(t | x)` is the survival function of the susceptible subgroup, and `1 - π(x)` is the cure fraction — the level at which survival asymptotes instead of decaying to zero.

The three models differ only in `S_u`, writing `λ(x) = exp(γ + X·δ)`:

| Model | Susceptible survival | Hazard shape |
|---|---|---|
| `LogNormalCureModel` | `Φ(-z)`, where `z = (log t - (γ + X·δ)) / σ` | rises then falls |
| `WeibullCureModel` | `exp(-(t / λ(x))^shape)` | monotone |
| `LogLogisticCureModel` | `1 / (1 + (t / λ(x))^shape)` | rises then falls, heavier tail |

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `alpha` | Intercept for susceptibility logit (at mean covariates) | `Normal(μ=0, σ=1)` |
| `beta_cure` | Covariate effects on susceptibility logit | `Normal(μ=0, σ=3)` |
| `gamma` | Intercept for the timing sub-model (at mean covariates) | `Normal(μ=0, σ=1)` |
| `delta` | Covariate effects on the timing sub-model | `Normal(μ=0, σ=2)` |
| `sigma` — log-normal only | Spread of log-event times among susceptibles | `HalfNormal(σ=1)` |
| `shape` — Weibull, log-logistic | Timing shape | `Gamma(α=5, β=2)` |

The timing shape is named `shape` rather than `alpha` to avoid collision with the cure sub-model intercept `alpha`.

> [!NOTE]
> The cure models centre covariates internally before sampling, which is what makes them sample reliably. The intercepts are reported back-transformed to their original meaning (the value at `X = 0`), so coefficients are interpreted exactly as before. Only the *priors* on `alpha` and `gamma` change meaning: they now describe the intercept at mean covariates — the better-conditioned and more weakly-informative choice.

```python
from bayes_survival import LogNormalCureModel

model = LogNormalCureModel()
model.fit(X_train, t_train, event_train, draws=1000, tune=1000, chains=4)

# Mixture survival function — plateaus at 1 - π rather than decaying to zero
pred = model.predict_survival_function(X_test, times=np.linspace(0.1, 36, 200))

# Posterior estimate of P(cured | x) = 1 - π(x)
cure = model.predict_cure_probability(X_test)
cure.mean, cure.hdi_lower, cure.hdi_upper       # each (n_obs,)

# Posterior predictive event times; cured individuals receive np.inf
model.sample_predicted_event_times(X_test)      # (n_samples, n_obs)
```

## Model comparison

`fit()` attaches a pointwise `log_likelihood` group by default, so ArviZ's comparison tools work directly. The Cox model's likelihood is summed from its internal long-format rows back to one value per subject, so its scores are on the same scale as the AFT and cure models:

```python
import arviz as az

weibull = WeibullAFTModel().fit(X, t, event)
cox = PiecewiseCoxPHModel(n_intervals=10).fit(X, t, event)

az.loo(weibull.idata)
az.compare({"weibull_aft": weibull.idata, "piecewise_cox": cox.idata})
```

Pass `fit(..., log_likelihood=False)` to skip the extra computation.

[`notebooks/Model_Comparison_Introduction.ipynb`](notebooks/Model_Comparison_Introduction.ipynb) walks through the full workflow — reading every column of the comparison table, spotting a fit whose diagnostics make it untrustworthy, and watching the ranking change when the data changes. The mechanics that make the scores comparable across model families are in [`notes/Model_Comparison.md`](notes/Model_Comparison.md).

## Held-out evaluation

`az.compare` answers *"which model fits the data I have?"*. `bayes_survival.metrics` answers *"does it work on subjects it never saw?"* — a different question, and the two can disagree.

Every function takes the same pair: a predicted survival matrix of shape `(n_obs, n_times)` and the time grid it was evaluated on. One `predict_survival_function` call therefore feeds all of them.

| Function | Returns | |
|---|---|---|
| `evaluate` | every metric below as one dict, ready to become a `DataFrame` row | |
| `harrell_concordance` | Harrell's C — re-exported from lifelines, not reimplemented | higher is better |
| `antolini_concordance` | Antolini's time-dependent C-td | higher is better |
| `ipcw_brier` | IPCW Brier score at one horizon | lower is better |
| `calibration_table` | predicted vs Kaplan-Meier-observed risk, by risk bin | monotone, near-diagonal |
| `censoring_distribution` | `Ghat(u)`, the estimate behind the IPCW weights | |

```python
from bayes_survival import WeibullAFTModel, evaluate, calibration_table
import numpy as np

# Put the horizons into the grid: np.interp is exact at knots, so this makes the
# Brier scores exact rather than interpolated.
horizons = [30.0, 180.0, 365.0]
grid = np.union1d(np.linspace(1.0, 730.0, 200), horizons)

model = WeibullAFTModel().fit(X_train, t_train, event_train)
pred = model.predict_survival_function(X_test, times=grid)

evaluate(pred, t_test, event_test, horizons=horizons,
         train=(t_train, event_train), name="weibull_aft")
# {'model': 'weibull_aft', 'C-index': ..., 'Antolini C-td': ...,
#  'Brier@30d': ..., 'Brier@180d': ..., 'Brier@365d': ...}

calibration_table(pred.mean, pred.times, t_test, event_test, horizon=180.0)
```

`train=(t_train, event_train)` is required and deliberately has no default: the censoring distribution behind the IPCW weights must be estimated on the *training* fold, and letting it fall back to the fold being scored would produce a subtly wrong Brier score with nothing to indicate it.

Two things worth knowing before reading the numbers. Concordance is a *ranking* measure and is structurally blind to miscalibration — a model can order subjects perfectly while every predicted probability is badly wrong, which is what `calibration_table` is for. And the Brier score has a strong base-rate floor, so read it against a covariate-free Kaplan-Meier baseline rather than against zero.

Antolini's C-td, not Harrell's C, is the metric the published survival benchmarks report; the two are different quantities and are not interchangeable. [`notebooks/SUPPORT2_AFT_Comparison.ipynb`](notebooks/SUPPORT2_AFT_Comparison.ipynb) puts all of this to work on real data, reproducing a published benchmark result end to end.

## Design

- `BaseSurvivalModel` — abstract base providing `fit`, `predict_survival_function`, `survival_probability`, `conditional_event_probability`, and `sample_predicted_event_times`
- Subclasses declare a `default_priors` class attribute; users can override any prior at construction time
- `build_model` uses `pm.Data` containers so `pm.set_data` can swap in new observations at predict time without rebuilding the graph
- `sample_predicted_event_times` calls `pm.sample_posterior_predictive` with `upper=inf` to draw uncensored event times from the posterior predictive distribution
- Every `fit()` validates its inputs up front — `t` must be strictly positive (the nonparametric estimators allow `t = 0`), `event` must be binary, and shapes must agree
- Every method that draws random samples accepts `random_seed` for reproducible output

## Future Work

Additional mixture cure models:

- Use [pymc-BART](https://www.pymc.io/projects/bart/en/latest/) as the classifier to allow for non-parametric modelling.
- Shared cure mixture interface so different classifying and timing components can be mixed freely.

## License

[MIT](LICENSE)
