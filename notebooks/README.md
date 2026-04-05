# Notebooks

Recommended reading order below. Each notebook builds on concepts from the previous ones.

---

## 1. [Bayes_Survival_Intro.ipynb](Bayes_Survival_Intro.ipynb)

**Start here.** A conceptual and mathematical primer covering:

- The Bayesian vs frequentist worldview — priors, posteriors, HDI vs confidence intervals
- Core survival analysis concepts: censoring, the survival function $S(t)$, and the hazard function $h(t)$
- How Bayesian inference applies to survival models, including the censored likelihood
- A minimal PyMC Weibull AFT model showing how priors and the censored likelihood are structured
- When to prefer Bayesian vs frequentist methods

No model fitting is performed here. The goal is to build intuition before the model-specific notebooks.

---

## 2. [KM_introduction.ipynb](KM_introduction.ipynb)

**Bayesian Kaplan-Meier — Beta-Binomial conjugate model.**

- Reviews the frequentist Kaplan-Meier estimator and its step-function survival estimate
- Introduces the Beta-Binomial conjugate model: a Beta prior on the conditional hazard at each event time, updated analytically — no MCMC required
- Shows why the Bayesian estimate declines faster than the KM curve and where the difference is largest (late time points, small risk sets)
- Explores how prior strength ($\alpha$, $\beta$) controls shrinkage, from near-MLE to strongly informative

---

## 3. [NA_introduction.ipynb](NA_introduction.ipynb)

**Bayesian Nelson-Aalen — Gamma-Poisson conjugate model.**

- Reviews the frequentist Nelson-Aalen estimator of the cumulative hazard $H(t)$
- Introduces the Gamma-Poisson conjugate model: a Gamma prior on the hazard increment at each event time, updated analytically — no MCMC required
- Compares cumulative hazard and survival estimates against the frequentist NA and KM curves
- Explores symmetric and asymmetric priors, showing how directional prior beliefs shift the hazard estimate

---

## 4. [AFT_Introduction.ipynb](AFT_Introduction.ipynb)

**Weibull AFT model — Bayesian (MCMC) vs frequentist (MLE).**

- Fits a Weibull Accelerated Failure Time model to the leukemia dataset using both `lifelines.WeibullAFTFitter` and `bayes_survival.WeibullAFTModel`
- Compares log-scale coefficients and shape parameter estimates side by side
- Shows posterior trace and density plots via ArviZ
- Compares predicted survival curves for clinically interpretable covariate profiles, highlighting the Bayesian HDI envelope vs the frequentist point estimate
- Demonstrates that both approaches recover near-identical coefficients on this dataset, while the Bayesian model provides richer uncertainty quantification

---

## 5. [CoxPH_Introduction.ipynb](CoxPH_Introduction.ipynb)

**Piecewise-constant Bayesian Cox PH model — hazard-scale regression.**

- Introduces the proportional hazards model $h(t \mid x) = h_0(t) \cdot \exp(X\beta)$ and contrasts it with AFT models, which act on the time scale rather than the hazard scale
- Derives the Poisson likelihood equivalence used to fit piecewise-constant Cox models and explains the data expansion to long format
- Shows how a Gaussian Random Walk prior on $\log h_0$ encodes smoothness across intervals without fixing the shape of the baseline hazard
- Fits `bayes_survival.PiecewiseCoxPHModel` and `lifelines.CoxPHFitter` to the leukemia dataset; compares log-hazard coefficients and survival curves
- Demonstrates sensitivity to the number of intervals and the GRW step-size prior

---

## 6. [Cure_Introduction.ipynb](Cure_Introduction.ipynb)

**Mixture cure model — covariate-dependent cure fractions.**

- Introduces the mixture cure model: $S_{\text{mix}}(t \mid x) = \pi(x) \cdot S_u(t \mid x) + (1 - \pi(x))$, where the survival curve plateaus above zero for subjects who will never experience the event
- Uses a synthetic e-commerce returns dataset with known true parameters for parameter recovery validation
- Compares `lifelines.MixtureCureFitter` (scalar cure fraction, no covariates) against `bayes_survival.LogNormalCureModel` (logistic regression sub-model for per-subject cure probability)
- Shows how the Bayesian model learns covariate-specific cure fractions and survival asymptotes that the frequentist model cannot capture
- Explains why `pm.Potential` is required instead of `pm.Censored` for the mixture likelihood
