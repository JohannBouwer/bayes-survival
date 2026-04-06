# PyMC / Probabilistic Programming

### How PyMC distributions work under the hood 
Built-in distributions provide `logp`, `logcdf`, and the Jacobian of these, enabling gradient-based MCMC. Any custom likelihood is possible via pm.Potential — the built-in distributions are convenience wrappers for common cases.

### `pm.Censored` vs `pm.Potential`
`pm.Censored` wraps a single distribution and automatically switches between log-PDF (observed) and log-survival (censored). pm.Potential injects arbitrary values into the log-probability, needed when the likelihood doesn't correspond to any single distribution (e.g., mixture cure models), or one of the predefined pymc distributions.

### `pt.switch` for branching likelihoods
PyTensor equivalent of np.where. Required when different observations contribute different likelihood terms (e.g., event vs censored in a cure model). Evaluates both branches for every observation, then selects element-wise.

### Conjugate Models and Analytical Inference

 - Conjugate is when the selection of the prior distribution and the likelihood results in a closed form posterior. This means no MCMC sampling is required and model fitting is fast.
 - Beta-Binomial conjugacy (Bayesian KM) $\rightarrow$ Hazard at each event time is a probability ∈ [0,1], so Beta is the natural prior. Posterior closes analytically: $Beta(α + d_j, β + n_j - d_j)$. No MCMC needed when hyperparameters are fixed.
 - Prior pseudo-observation interpretation $\rightarrow$ In Beta(α,β): α = pseudo-deaths, β = pseudo-survivals. Total weight α+β controls prior strength independently of prior mean α/(α+β). Beta(1,1) looks uninformative but injects 2 pseudo-observations per time point ( strong for small sample numbers).