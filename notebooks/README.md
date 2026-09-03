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

## 6. [Hierarchical_AFT_Introduction.ipynb](Hierarchical_AFT_Introduction.ipynb)

**Hierarchical Weibull AFT — partial pooling over covariate groups.**

- Motivates hierarchical priors: when covariates form natural groups (e.g. product categories, day-of-week dummies), sharing a hyper-prior lets sparse groups borrow strength from data-rich ones
- Fits a flat `WeibullAFTModel` as a baseline, then fits `HierarchicalWeibullAFTModel` with two independent hierarchies (`product_type` and `day_of_week`) on a synthetic e-commerce returns dataset
- Inspects hyper-prior posteriors: `mu_group` captures the average group signal; `sigma_group` quantifies how much members within the group vary
- Demonstrates partial pooling: the sports category (n≈50) has its HDI width shrunk by borrowing from the other product types
- Compares flat vs hierarchical survival predictions for four representative customer profiles, showing tighter uncertainty for sparse groups under the hierarchical model

---

## 7. [Cure_Introduction.ipynb](Cure_Introduction.ipynb)

**Mixture cure model — covariate-dependent cure fractions.**

- Introduces the mixture cure model: $S_{\text{mix}}(t \mid x) = \pi(x) \cdot S_u(t \mid x) + (1 - \pi(x))$, where the survival curve plateaus above zero for subjects who will never experience the event
- Uses a synthetic e-commerce returns dataset with known true parameters for parameter recovery validation
- Compares `lifelines.MixtureCureFitter` (scalar cure fraction, no covariates) against `bayes_survival.LogLogisticCureModel` (logistic regression sub-model for per-subject cure probability)
- Shows how the Bayesian model learns covariate-specific cure fractions and survival asymptotes that the frequentist model cannot capture
- Explains why `pm.Potential` is required instead of `pm.Censored` for the mixture likelihood

---

## 8. [Model_Comparison_Introduction.ipynb](Model_Comparison_Introduction.ipynb)

**Comparing models across families with `az.loo` and `az.compare`.**

- Introduces leave-one-out cross-validation: `elpd_loo` as a relative score, `p_loo` as effective parameters, and Pareto $k$ as the reliability diagnostic
- Fits all seven models — three AFT, piecewise Cox, and three mixture cure — to one synthetic dataset and ranks them with `az.compare`, walking through every column of the output
- Shows why `dse` matters more than `elpd_diff`: the top two models differ by about one standard error of their difference, so LOO identifies the right model *family* rather than a single winner
- Works through the diagnostics that have to pass before a ranking means anything — divergences, `r_hat`, and Pareto $k$ — and what to try when they do not
- Reruns the whole comparison on the same data-generating process with the cure fraction removed, and the ranking changes — demonstrating that LOO responds to the data rather than rewarding complexity
- Closes with the four conditions that make a comparison valid, none of which `az.compare` checks; the mechanics are in [notes/Model_Comparison.md](../notes/Model_Comparison.md)

---

## 9. [SUPPORT2_AFT_Comparison.ipynb](SUPPORT2_AFT_Comparison.ipynb)

**Real data, end to end — model selection and held-out validation on a published benchmark.**

Every notebook above runs on synthetic data or a small textbook dataset, which can only show that a
model recovers what was put into it. This one runs on **SUPPORT2** — 9,105 seriously ill hospitalised
adults, 68% mortality, follow-up 3–2,029 days — so the results can be checked against numbers other
people have published.

- Builds a leakage-free modelling table with [`scripts/prepare_support2.py`](../scripts/prepare_support2.py), which downloads SUPPORT2 and documents why each excluded column is excluded — the study ships its own predicted survival, physicians' prognoses, and post-baseline outcomes, all of which would flatter the C-index meaninglessly
- Fits the three flat parametric AFT families under matched priors and selects between them with `az.compare`, checking divergences, `r_hat`, and Pareto $k$ before reading the ranking
- Validates the winner three independent ways: against `lifelines` maximum likelihood estimates, against the published literature, and against Kaplan-Meier within predicted-risk quintiles
- **Reproduces the published benchmark**: on the same 14 covariates and the same concordance definition used by Kvamme, Borgan & Scheel (2019), [JMLR 20(129)](https://jmlr.org/papers/volume20/18-424/18-424.pdf), the pipeline recovers their classical-Cox C-td of 0.598 to within 0.002
- Uses the held-out metrics in [`bayes_survival.metrics`](../src/bayes_survival/metrics.py) — Antolini's $C^{td}$, IPCW Brier, and a Kaplan-Meier calibration table — and shows where the benchmark covariate set systematically miscalibrates by disease group, which concordance alone is blind to

The notebook fetches and builds its own data on first run; no manual setup is required.
