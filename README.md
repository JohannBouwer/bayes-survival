# bayes-survival

> [!NOTE]
> **Heads up:** This is a project I completed for my own development / learning. The models work and have been tested, but no claims are made about production-readiness, computational efficiency, or suitability for any particular use case.

Bayesian survival analysis models built on [PyMC](https://www.pymc.io/), with comparisons against [lifelines](https://lifelines.readthedocs.io/) frequentist equivalents. Inspired by [pymc-survival](https://github.com/pymc-labs/pymc-survival).

## Table of Contents

- [Installation](#installation)
  - [Core library](#core-library)
  - [With notebook dependencies](#with-notebook-dependencies)
  - [With dev dependencies](#with-dev-dependencies)
  - [With all optional dependencies](#with-all-optional-dependencies)
- [Models](#models)
  - [Nonparametric](#nonparametric)
    - [`KaplanMeierModel`](#kaplanmeiermodel)
    - [`NelsonAalenModel`](#nelsonalenmodel)
  - [Accelerated Failure Time (AFT)](#accelerated-failure-time-aft)
    - [`WeibullAFTModel`](#weibullaftmodel)
    - [`LogNormalAFTModel`](#lognormalaftmodel)
    - [`LogLogisticAFTModel`](#loglogisticaftmodel)
  - [Hierarchical Accelerated Failure Time (AFT)](#hierarchical-accelerated-failure-time-aft)
    - [`HierarchicalWeibullAFTModel`](#hierarchicalweibullaftmodel)
    - [`HierarchicalLogNormalAFTModel`](#hierarchicallognormalaftmodel)
    - [`HierarchicalLogLogisticAFTModel`](#hierarchicalloglogisticaftmodel)
  - [Cox Proportional Hazards](#cox-proportional-hazards)
    - [`PiecewiseCoxPHModel`](#piecewisecoxphmodel)
  - [Mixture Cure Models](#mixture-cure-models)
    - [`LogNormalCureModel`](#lognormalcuremodel)
    - [`WeibullCureModel`](#weibullcuremodel)
    - [`LogLogisticCureModel`](#loglogisticcuremodel)
- [Design](#design)
- [Future Work](#future-work)
  - [Additional Mixture Cure Models](#additional-mixture-cure-models)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Core library

```bash
git clone https://github.com/Johann-FullHD/bayes-survival.git
cd bayes-survival
uv sync
```

### With notebook dependencies

Installs JupyterLab, ipykernel, ipywidgets, and watermark alongside the core library:

```bash
uv sync --extra notebook
```

### With dev dependencies

Installs pytest and ruff alongside the core library:

```bash
uv sync --extra dev
```

### With all optional dependencies

```bash
uv sync --all-extras
```

> [!IMPORTANT]
> Do not install dependencies one by one, each `uv sync --option` will overwrite the previous one


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

```python
from bayes_survival.nonparametric import KaplanMeierModel
import numpy as np

km = KaplanMeierModel()
km.fit(t, event)

# Posterior survival function with 94% HDI
pred = km.predict_survival_function(times=np.linspace(0, 36, 200))
pred.mean       # (1, n_times)
pred.hdi_lower  # (1, n_times)
pred.hdi_upper  # (1, n_times)

# Analytical posterior mean (closed form, no sampling)
times, S_mean = km.posterior_mean_survival

# Survival probability at a single time
km.survival_probability(t=12.0)

# Raw posterior samples
km.sample_posterior_survival(times, n_samples=2000)  # (n_samples, n_times)
```

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
from bayes_survival.nonparametric import NelsonAalenModel
import numpy as np

na = NelsonAalenModel()
na.fit(t, event)

# Posterior cumulative hazard with 94% HDI
pred = na.predict_cumulative_hazard(times=np.linspace(0, 36, 200))
pred.mean       # (1, n_times)

# Posterior survival function (via exp(-H(t)))
pred = na.predict_survival_function(times=np.linspace(0, 36, 200))

# Analytical posterior mean cumulative hazard (closed form)
times, H_mean = na.posterior_mean_cumulative_hazard

# Analytical posterior mean survival
times, S_mean = na.posterior_mean_survival
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
from bayes_survival.survival_models.aft import WeibullAFTModel, LogNormalAFTModel, LogLogisticAFTModel
import numpy as np

# Inspect default priors before fitting
WeibullAFTModel.default_priors
# {'alpha': (Gamma, {'alpha': 5, 'beta': 2}), 'beta': (Normal, {'mu': 0, 'sigma': 5})}

# Fit with defaults
model = WeibullAFTModel()
model.fit(X_train, t_train, event_train, draws=1000, tune=1000, chains=4)

# Override a prior
model = WeibullAFTModel(priors={"alpha": (pm.HalfNormal, {"sigma": 1})})

# Survival function: returns mean + 94% HDI over a time grid
pred = model.predict_survival_function(X_test, times=np.linspace(0.1, 36, 200))
pred.mean        # (n_obs, n_times)
pred.hdi_lower   # (n_obs, n_times)
pred.hdi_upper   # (n_obs, n_times)

# Survival probability at a single time point
model.survival_probability(X_test, t=12.0)

# Conditional probability of event before T given survival to t
model.conditional_event_probability(X_test, t=6.0, T=24.0)

# Posterior predictive event time distribution (full distribution per individual)
samples = model.sample_predicted_event_times(X_test)  # (n_samples, n_obs)
```

### Hierarchical Accelerated Failure Time (AFT)

Hierarchical variants of each AFT model. Covariates belonging to a
`HierarchySpec` group are drawn from `Normal(mu_group, sigma_group)`, where
the hyper-parameters are themselves given priors — enabling **partial pooling**
across the group. Sparse groups borrow strength from data-rich groups instead
of being estimated in isolation.

`fit()` requires a `pd.DataFrame` so covariate names can be resolved.
Prediction methods accept either a DataFrame or a plain NumPy array (columns in
the same order as the training DataFrame).

#### `HierarchicalWeibullAFTModel`

Extends `WeibullAFTModel` with hierarchical priors on selected covariate groups.

```
S(t | x) = exp( -(t / exp(Xβ))^α )
Grouped:   β_j ~ Normal(mu_group, sigma_group)
```

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `beta` (intercept + ungrouped) | Flat prior | `Normal(μ=0, σ=5)` |
| `mu_{group}` | Hyper-prior on group mean | `Normal(μ=0, σ=1)` |
| `sigma_{group}` | Hyper-prior on group std | `HalfNormal(σ=1)` |
| `alpha` | Weibull shape | `Gamma(α=5, β=2)` |

#### `HierarchicalLogNormalAFTModel`

Extends `LogNormalAFTModel` with hierarchical priors on selected covariate groups.

```
S(t | x) = Φ((Xβ - log(t)) / σ)
Grouped:   β_j ~ Normal(mu_group, sigma_group)
```

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `beta` (intercept + ungrouped) | Flat prior | `Normal(μ=0, σ=5)` |
| `mu_{group}` | Hyper-prior on group mean | `Normal(μ=0, σ=1)` |
| `sigma_{group}` | Hyper-prior on group std | `HalfNormal(σ=1)` |
| `sigma` | Spread of log-event times | `Gamma(α=5, β=2)` |

#### `HierarchicalLogLogisticAFTModel`

Extends `LogLogisticAFTModel` with hierarchical priors on selected covariate groups.

```
S(t | x) = 1 / (1 + (t / exp(Xβ))^α)
Grouped:   β_j ~ Normal(mu_group, sigma_group)
```

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `beta` (intercept + ungrouped) | Flat prior | `Normal(μ=0, σ=5)` |
| `mu_{group}` | Hyper-prior on group mean | `Normal(μ=0, σ=1)` |
| `sigma_{group}` | Hyper-prior on group std | `HalfNormal(σ=1)` |
| `alpha` | Shape — controls tail heaviness and hazard peak | `Gamma(α=5, β=2)` |

```python
from bayes_survival.survival_models.hierarchical_aft import (
    HierarchicalWeibullAFTModel,
    HierarchicalLogNormalAFTModel,
    HierarchicalLogLogisticAFTModel,
)
from bayes_survival.survival_models.base import HierarchySpec
import numpy as np

# Define covariate groups that should share a hyper-prior
hierarchies = [
    HierarchySpec(name="product_type", covariate_names=["electronics", "clothing", "books"]),
    HierarchySpec(name="day_of_week", covariate_names=["mon", "tue", "wed", "thu"]),
]

model = HierarchicalWeibullAFTModel(hierarchies=hierarchies)
model.fit(df_train, t_train, event_train, draws=1000, tune=1000, chains=4)

# Survival function: mean + 94% HDI
pred = model.predict_survival_function(df_test, times=np.linspace(0.1, 36, 200))
pred.mean        # (n_obs, n_times)
pred.hdi_lower   # (n_obs, n_times)
pred.hdi_upper   # (n_obs, n_times)

# Survival probability at a single time
model.survival_probability(df_test, t=12.0)

# Posterior predictive event times
samples = model.sample_predicted_event_times(df_test)  # (n_samples, n_obs)
```

### Cox Proportional Hazards

#### `PiecewiseCoxPHModel`

A piecewise-constant Bayesian Cox PH model. The hazard is:

```
h(t | x) = h_0(t) · exp(Xβ)
```

where `h_0(t)` is piecewise constant over `K` intervals and `log h_0` follows a Gaussian Random Walk across intervals (smoothness prior). No intercept is added to `β`; the baseline hazard absorbs it.

Fitting uses the Poisson likelihood equivalence: data are expanded to long format (one row per observation-interval pair while at risk) and event counts are modelled as Poisson with rate `h_k · exp(Xβ) · exposure`.

Cut points can be supplied explicitly or placed automatically at evenly-spaced quantiles of the observed event times.

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `log_baseline` | Log baseline hazard per interval (GRW) | `GaussianRandomWalk(sigma=grw_sigma)` |
| `grw_sigma` | Random-walk step-size (smoothness) | `HalfNormal(σ=1)` |
| `beta` | Log-hazard coefficients | `Normal(μ=0, σ=1)` |

```python
from bayes_survival.survival_models.cox_hazard import PiecewiseCoxPHModel
import numpy as np

# Automatic cut points at event-time quantiles
model = PiecewiseCoxPHModel(n_intervals=10)
model.fit(X_train, t_train, event_train, draws=1000, tune=1000, chains=4)

# Or supply explicit interior cut points
model = PiecewiseCoxPHModel(cuts=[6.0, 12.0, 24.0])

# Override priors
model = PiecewiseCoxPHModel(n_intervals=10, priors={"beta": (pm.Normal, {"mu": 0, "sigma": 0.5})})

# Survival function: mean + 94% HDI
pred = model.predict_survival_function(X_test, times=np.linspace(0.1, 36, 200))
pred.mean        # (n_obs, n_times)
pred.hdi_lower   # (n_obs, n_times)
pred.hdi_upper   # (n_obs, n_times)

# Survival probability at a single time point
model.survival_probability(X_test, t=12.0)

# Conditional probability of event before T given survival to t
model.conditional_event_probability(X_test, t=6.0, T=24.0)

# Posterior predictive event times via piecewise-exponential inverse CDF
samples = model.sample_predicted_event_times(X_test)  # (n_samples, n_obs)
```

### Mixture Cure Models

Some datasets contain individuals who will never experience the event — they are "cured". Standard AFT models cannot represent this: they assign non-zero hazard to all individuals at all future times, so the survival function eventually reaches zero for everyone.

A **mixture cure model** splits the population into two latent groups:

```
S_mix(t | x) = π(x) · S_u(t | x) + (1 - π(x))
```

where:
- `π(x) = sigmoid(α + X·β_cure)` — probability of being susceptible (not cured)
- `S_u(t | x)` — survival function for the susceptible subgroup
- `1 - π(x)` — cure fraction; survival asymptotes here instead of reaching zero

#### `LogNormalCureModel`

Log-normal timing distribution for the susceptible subgroup:

```
S_u(t | x) = Φ(-z),   z = (log(t) - (γ + X·δ)) / σ
```

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `alpha` | Intercept for susceptibility logit | `Normal(μ=0, σ=1)` |
| `beta_cure` | Covariate effects on susceptibility logit | `Normal(μ=0, σ=3)` |
| `gamma` | Intercept for log-normal log-mean | `Normal(μ=0, σ=1)` |
| `delta` | Covariate effects on log-mean | `Normal(μ=0, σ=2)` |
| `sigma` | Spread of log-event times (susceptibles) | `HalfNormal(σ=1)` |

```python
from bayes_survival.survival_models.cure import LogNormalCureModel
import numpy as np

model = LogNormalCureModel()
model.fit(X_train, t_train, event_train, draws=1000, tune=1000, chains=4)

# Mixture survival function (accounts for cure fraction)
pred = model.predict_survival_function(X_test, times=np.linspace(0.1, 36, 200))
pred.mean        # (n_obs, n_times) — plateaus at 1 - π for cured individuals
pred.hdi_lower   # (n_obs, n_times)
pred.hdi_upper   # (n_obs, n_times)

# Posterior estimate of P(cured | x) = 1 - π(x)
cure = model.predict_cure_probability(X_test)
cure.mean        # (n_obs,)
cure.hdi_lower   # (n_obs,)
cure.hdi_upper   # (n_obs,)

# Posterior predictive event times; cured individuals receive np.inf
samples = model.sample_predicted_event_times(X_test)  # (n_samples, n_obs)
```

#### `WeibullCureModel`

Weibull timing distribution for the susceptible subgroup:

```
S_u(t | x) = exp(-(t / λ(x))^shape),   λ(x) = exp(γ + X·δ)
```

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `alpha` | Intercept for susceptibility logit | `Normal(μ=0, σ=1)` |
| `beta_cure` | Covariate effects on susceptibility logit | `Normal(μ=0, σ=3)` |
| `gamma` | Intercept for Weibull log-scale | `Normal(μ=0, σ=1)` |
| `delta` | Covariate effects on Weibull log-scale | `Normal(μ=0, σ=2)` |
| `shape` | Weibull shape (shape > 1: increasing hazard) | `Gamma(α=5, β=2)` |

Note: the Weibull shape is named `shape` rather than `alpha` to avoid collision with the cure sub-model intercept `alpha`.

```python
from bayes_survival.survival_models.cure import WeibullCureModel
import numpy as np

model = WeibullCureModel()
model.fit(X_train, t_train, event_train, draws=1000, tune=1000, chains=4)

# Mixture survival function (accounts for cure fraction)
pred = model.predict_survival_function(X_test, times=np.linspace(0.1, 36, 200))
pred.mean        # (n_obs, n_times) — plateaus at 1 - π for cured individuals
pred.hdi_lower   # (n_obs, n_times)
pred.hdi_upper   # (n_obs, n_times)

# Posterior estimate of P(cured | x) = 1 - π(x)
cure = model.predict_cure_probability(X_test)
cure.mean        # (n_obs,)
cure.hdi_lower   # (n_obs,)
cure.hdi_upper   # (n_obs,)

# Posterior predictive event times; cured individuals receive np.inf
samples = model.sample_predicted_event_times(X_test)  # (n_samples, n_obs)
```

#### `LogLogisticCureModel`

Log-logistic timing distribution for the susceptible subgroup:

```
S_u(t | x) = 1 / (1 + (t / λ(x))^shape),   λ(x) = exp(γ + X·δ)
```

The log-logistic hazard is non-monotonic (rises then falls), making this model appropriate when susceptible subjects face a peak event rate at some intermediate time rather than a monotonically increasing or decreasing risk.

| Parameter | Role | Default prior |
|-----------|------|---------------|
| `alpha` | Intercept for susceptibility logit | `Normal(μ=0, σ=1)` |
| `beta_cure` | Covariate effects on susceptibility logit | `Normal(μ=0, σ=3)` |
| `gamma` | Intercept for log-logistic log-scale | `Normal(μ=0, σ=1)` |
| `delta` | Covariate effects on log-scale | `Normal(μ=0, σ=2)` |
| `shape` | Log-logistic shape (controls tail heaviness and hazard peak) | `Gamma(α=5, β=2)` |

Note: the shape parameter is named `shape` rather than `alpha` to avoid collision with the cure sub-model intercept `alpha`.

```python
from bayes_survival.survival_models.cure import LogLogisticCureModel
import numpy as np

model = LogLogisticCureModel()
model.fit(X_train, t_train, event_train, draws=1000, tune=1000, chains=4)

# Mixture survival function (accounts for cure fraction)
pred = model.predict_survival_function(X_test, times=np.linspace(0.1, 36, 200))
pred.mean        # (n_obs, n_times) — plateaus at 1 - π for cured individuals
pred.hdi_lower   # (n_obs, n_times)
pred.hdi_upper   # (n_obs, n_times)

# Posterior estimate of P(cured | x) = 1 - π(x)
cure = model.predict_cure_probability(X_test)
cure.mean        # (n_obs,)
cure.hdi_lower   # (n_obs,)
cure.hdi_upper   # (n_obs,)

# Posterior predictive event times; cured individuals receive np.inf
samples = model.sample_predicted_event_times(X_test)  # (n_samples, n_obs)
```

## Design

- `BaseSurvivalModel` — abstract base providing `fit`, `predict_survival_function`, `survival_probability`, `conditional_event_probability`, and `sample_predicted_event_times`
- Subclasses declare a `default_priors` class attribute; users can override any prior at construction time
- `build_model` uses `pm.Data` containers so `pm.set_data` can swap in new observations at predict time without rebuilding the graph
- `sample_predicted_event_times` calls `pm.sample_posterior_predictive` with `upper=inf` to draw uncensored event times from the posterior predictive distribution

## Future Work

### Additional Mixture Cure Models

- Use [pymc-BART](https://www.pymc.io/projects/bart/en/latest/) as the classifier to allow for non-parametric modelling.
- Shared cure mixture interface so different classifying and timing components can be mixed freely.
- Hierarchical cure models.
