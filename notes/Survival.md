# Survival Models

### Right-censoring mechanics 
Observed events contribute `log f(t)` to the likelihood; censored observations contribute `log S(t)`. This is the fundamental distinction survival models handle that ordinary regression cannot, censored data contributes to the likelihood differently. 

### AFT vs Cox PH (what covariates act on) 
AFT rescales time ($\exp(Xβ)$ stretches/compresses when events happen). Cox PH rescales hazard ($\exp(Xβ)$ multiplies how fast events accumulate). 
- The partial likelihood trick that makes frequentist Cox elegant is incompatible with Bayesian inference because it marginalises out the baseline hazard, and Bayes needs a proper likelihood over all unknowns.

### Conditional event probability 
$P(\text{event before T} | \text{survived to t}) = 1 - S(T)/S(t)$ requires computing the ratio per posterior sample, then aggregating. Computing the ratio on already-aggregated means/HDIs gives the wrong answer.

### Survival curves vs posterior predictive event times 
"what is the probability of surviving past time t?" is a deterministic function of parameters, while `sample_posterior_predictive` answers "when will the event occur?" (a draw from the likelihood). 