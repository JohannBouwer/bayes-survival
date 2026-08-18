from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import ndtr, expit

from .base import BaseSurvivalModel, PriorSpec


@dataclass
class CurePrediction:
    """Posterior summary of P(cured | x) = 1 - π(x) for each observation."""

    mean: np.ndarray  # (n_obs,)
    hdi_lower: np.ndarray  # (n_obs,)
    hdi_upper: np.ndarray  # (n_obs,)


def _lift_obs_loglik(idata: az.InferenceData) -> None:
    """Move the ``obs_loglik`` Deterministic into a ``log_likelihood`` group.

    The mixture cure likelihood is expressed with ``pm.Potential``, which is not
    an observed RV — ``pm.compute_log_likelihood`` therefore produces nothing at
    all for these models (silently: no group, no error). ``build_model`` instead
    records the pointwise log-likelihood as a Deterministic, and this promotes it
    to the group ``az.loo`` / ``az.compare`` expect.

    Dropping it from ``posterior`` also keeps it out of ``az.summary`` and trace
    plots, where an (n_obs,) variable would otherwise render hundreds of rows.
    """
    if idata is None or "obs_loglik" not in idata.posterior:
        return

    da = idata.posterior["obs_loglik"]
    extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
    renamed = da.rename({d: f"obs_dim_{i}" for i, d in enumerate(extra_dims)})

    idata.add_groups({"log_likelihood": renamed.to_dataset(name="obs")})
    idata.posterior = idata.posterior.drop_vars("obs_loglik")


def _cure_sub_model(model, X_centred, x_bar, n_features):
    """Susceptibility sub-model on centred covariates.

    Covariates are centred so that the sampled intercept is the logit of
    susceptibility *at mean covariates*. With uncentred covariates the intercept
    and the slopes are strongly correlated (condition number of XᵀX on the
    reference dataset: 92.2 uncentred vs 3.4 centred), producing a long diagonal
    ridge that the sampler has to traverse.

    The reported ``alpha`` is back-transformed to its original meaning — the
    intercept at X = 0 — so the prediction paths, which combine ``alpha`` with
    raw covariates, need no adjustment:

        logit π = α_c + (X − x̄)·β = (α_c − x̄·β) + X·β

    Returns ``(logit_pi, beta_cure)``.
    """
    alpha_c = model._prior("alpha", var_name="alpha_centred")
    beta_cure = model._prior("beta_cure", shape=n_features)

    logit_pi = alpha_c + pt.dot(X_centred, beta_cure)
    pm.Deterministic("alpha", alpha_c - pt.dot(x_bar, beta_cure))
    pm.Deterministic("pi", pt.sigmoid(logit_pi))
    return logit_pi, beta_cure


def _timing_linpred(model, X_centred, x_bar, n_features):
    """Timing sub-model linear predictor on centred covariates.

    Same reparameterisation as :func:`_cure_sub_model`: ``gamma_centred`` is
    sampled, ``gamma`` is reported on the original scale.

    Returns ``(linear_predictor, delta)``.
    """
    gamma_c = model._prior("gamma", var_name="gamma_centred")
    delta = model._prior("delta", shape=n_features)

    lin = gamma_c + pt.dot(X_centred, delta)
    pm.Deterministic("gamma", gamma_c - pt.dot(x_bar, delta))
    return lin, delta


def _register_mixture_likelihood(logit_pi, log_S, log_f, event_data):
    """Register the mixture cure likelihood as a Deterministic and a Potential.

    Works on the logit scale throughout, via ``log sigmoid(z) = -softplus(-z)``.
    That form cannot underflow: at ``z = -800`` it returns ``-800``, where
    ``log(sigmoid(z))`` returns ``-inf``. The censored branch is a
    ``logaddexp`` rather than ``log(π·exp(log_S) + 1 − π)``, keeping precision
    when ``log_S`` is very negative.

    The pointwise log-likelihood is registered twice as *independent nodes* built
    from the same expression: a Deterministic (lifted into a ``log_likelihood``
    group after sampling, for az.loo/az.compare) and the Potential that
    contributes to the model logp.

    Do not write ``pm.Potential("obs", <the Deterministic variable>)`` — wrapping
    the Deterministic makes the nutpie backend, this library's default sampler,
    fail with ``KeyError: 'obs'``.
    """
    log_pi = -pt.softplus(-logit_pi)      # log sigmoid(z)   = log pi
    log_1m_pi = -pt.softplus(logit_pi)    # log sigmoid(-z)  = log (1 - pi)

    log_lik_event = log_pi + log_f
    log_lik_censored = pt.logaddexp(log_pi + log_S, log_1m_pi)

    loglik = pt.switch(pt.eq(event_data, 1.0), log_lik_event, log_lik_censored)
    pm.Deterministic("obs_loglik", loglik)
    pm.Potential("obs", loglik)


class LogNormalCureModel(BaseSurvivalModel):
    r"""Mixture cure survival model with log-normal timing distribution.

    Mixture survival function:
        S_mix(t | x) = π(x) · S_u(t | x) + (1 - π(x))

    where:
        π(x)       = sigmoid(\alpha + X·β_cure)   — P(susceptible | x)
        S_u(t | x) = Φ(-z)                   — log-normal survival among susceptibles
        z          = (log(t) - (\gamma + X·δ)) / \sigma

    Subjects in the "cured" fraction (probability 1 - π) never experience the event,
    so their mixture survival function asymptotes to 1 - π instead of zero.

    An intercept is included in each sub-model (\alpha for cure, \gamma for timing); users
    pass raw covariates only — no need to add a ones column.
    """

    default_priors: ClassVar[PriorSpec] = {
        "alpha": (pm.Normal, {"mu": 0, "sigma": 1}),
        "beta_cure": (pm.Normal, {"mu": 0, "sigma": 3}),
        "gamma": (pm.Normal, {"mu": 0, "sigma": 1}),
        "delta": (pm.Normal, {"mu": 0, "sigma": 2}),
        "sigma": (pm.HalfNormal, {"sigma": 1}),
    }

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        self._n_features = X.shape[1]
        x_bar = X.mean(axis=0)

        with pm.Model() as model:
            # Covariates are centred for sampling; see _cure_sub_model for why.
            X_data = pm.Data("X", (X - x_bar).astype(float))
            t_data = pm.Data("t_obs", t.astype(float))
            event_data = pm.Data("event", event.astype(float))

            # Cure sub-model: π(x) = sigmoid(\alpha + X·β_cure)
            logit_pi, beta_cure = _cure_sub_model(self, X_data, x_bar, X.shape[1])

            # Timing sub-model: log-normal with mean \gamma + X·δ on the log scale
            mu, delta = _timing_linpred(self, X_data, x_bar, X.shape[1])
            sigma = self._prior("sigma")

            # log S_u(t | x) = log(0.5 · erfc((log t - μ) / (\sigma√2)))
            log_S = pt.log(
                0.5 * pt.erfc((pt.log(t_data) - mu) / (sigma * pt.sqrt(2.0)))
            )

            log_f = pm.logp(pm.LogNormal.dist(mu=mu, sigma=sigma), t_data)

            _register_mixture_likelihood(logit_pi, log_S, log_f, event_data)

        return model

    def _attach_log_likelihood(self) -> None:
        _lift_obs_loglik(self.idata)

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = self._check_n_features(X)
        times = np.asarray(times, dtype=float)

        assert self.idata is not None
        posterior = self.idata.posterior

        alpha_s = posterior["alpha"].values  # (chains, draws)
        beta_cure_s = posterior["beta_cure"].values  # (chains, draws, n_features)
        gamma_s = posterior["gamma"].values  # (chains, draws)
        delta_s = posterior["delta"].values  # (chains, draws, n_features)
        sigma_s = posterior["sigma"].values  # (chains, draws)

        n_chains, n_draws, n_features = beta_cure_s.shape
        n_samples = n_chains * n_draws

        alpha_flat = alpha_s.reshape(n_samples)  # (n_samples,)
        beta_cure_flat = beta_cure_s.reshape(
            n_samples, n_features
        )  # (n_samples, n_features)
        gamma_flat = gamma_s.reshape(n_samples)  # (n_samples,)
        delta_flat = delta_s.reshape(n_samples, n_features)  # (n_samples, n_features)
        sigma_flat = sigma_s.reshape(n_samples)  # (n_samples,)

        # pi: (n_samples, n_obs)
        pi = expit(alpha_flat[:, np.newaxis] + beta_cure_flat @ X.T)

        # mu: (n_samples, n_obs)
        mu = gamma_flat[:, np.newaxis] + delta_flat @ X.T

        # z: (n_samples, n_obs, n_times)
        log_t = np.log(times)  # (n_times,)
        z = (log_t[np.newaxis, np.newaxis, :] - mu[:, :, np.newaxis]) / sigma_flat[
            :, np.newaxis, np.newaxis
        ]

        # S_u: log-normal survival, shape (n_samples, n_obs, n_times)
        S_u = ndtr(-z)

        # S_mix = π · S_u + (1 - π), broadcasting pi to (n_samples, n_obs, 1)
        return pi[:, :, np.newaxis] * S_u + (1.0 - pi[:, :, np.newaxis])

    def predict_cure_probability(
        self,
        X: np.ndarray,
        hdi_prob: float = 0.94,
    ) -> CurePrediction:
        """Posterior estimate of P(cured | x) = 1 - π(x) for each observation.

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        hdi_prob : credible interval width (default 0.94).

        Returns
        -------
        CurePrediction with arrays of shape (n_obs,).
        """
        self._check_fitted()
        X = self._check_n_features(X)

        assert self.idata is not None
        posterior = self.idata.posterior

        alpha_flat = posterior["alpha"].values.ravel()  # (n_samples,)
        beta_cure_flat = posterior["beta_cure"].values.reshape(
            -1, X.shape[1]
        )  # (n_samples, p)

        # cure_prob[s, i] = 1 - sigmoid(\alpha_s + x_i · β_s) = P(cured | x_i, θ_s)
        cure_prob = 1.0 - expit(
            alpha_flat[:, np.newaxis] + beta_cure_flat @ X.T
        )  # (n_samples, n_obs)

        mean = cure_prob.mean(axis=0)
        hdi = np.stack(
            [az.hdi(cure_prob[:, i], hdi_prob=hdi_prob) for i in range(X.shape[0])],
            axis=0,
        )  # (n_obs, 2)

        return CurePrediction(mean=mean, hdi_lower=hdi[:, 0], hdi_upper=hdi[:, 1])

    def sample_predicted_event_times(
        self,
        X: np.ndarray,
        return_idata: bool = False,
        random_seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Draw posterior-predictive event times from the mixture cure model.

        Susceptible subjects (drawn from Bernoulli(π)) get a log-normal event time;
        non-susceptibles receive ``np.inf`` (they never experience the event).

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        return_idata : must be False; InferenceData is not available for numpy samples.
        random_seed : seed for the predictive draws; pass an int for reproducibility.

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs).
        """
        if return_idata:
            raise NotImplementedError(
                "return_idata=True is not supported for LogNormalCureModel — "
                "samples are drawn in numpy and have no InferenceData wrapper."
            )
        rng = np.random.default_rng(random_seed)
        self._check_fitted()
        X = self._check_n_features(X)

        assert self.idata is not None
        posterior = self.idata.posterior

        alpha_s = posterior["alpha"].values
        beta_cure_s = posterior["beta_cure"].values
        gamma_s = posterior["gamma"].values
        delta_s = posterior["delta"].values
        sigma_s = posterior["sigma"].values

        n_chains, n_draws, n_features = beta_cure_s.shape
        n_samples = n_chains * n_draws

        alpha_flat = alpha_s.reshape(n_samples)
        beta_cure_flat = beta_cure_s.reshape(n_samples, n_features)
        gamma_flat = gamma_s.reshape(n_samples)
        delta_flat = delta_s.reshape(n_samples, n_features)
        sigma_flat = sigma_s.reshape(n_samples)

        pi = expit(
            alpha_flat[:, np.newaxis] + beta_cure_flat @ X.T
        )  # (n_samples, n_obs)
        mu = gamma_flat[:, np.newaxis] + delta_flat @ X.T  # (n_samples, n_obs)

        susceptible = rng.binomial(1, pi).astype(bool)  # (n_samples, n_obs)
        t_latent = rng.lognormal(
            mu, sigma_flat[:, np.newaxis], size=mu.shape
        )  # (n_samples, n_obs)

        return np.where(susceptible, t_latent, np.inf)


class WeibullCureModel(BaseSurvivalModel):
    """Mixture cure survival model with Weibull timing distribution.

    Mixture survival function:
        S_mix(t | x) = π(x) · S_u(t | x) + (1 - π(x))

    where:
        π(x)       = sigmoid(α + X·β_cure)          — P(susceptible | x)
        S_u(t | x) = exp(-(t / λ(x))^shape)         — Weibull survival among susceptibles
        λ(x)       = exp(γ + X·δ)                   — Weibull scale (log link)

    Subjects in the "cured" fraction (probability 1 - π) never experience the event,
    so their mixture survival function asymptotes to 1 - π instead of zero.

    An intercept is included in each sub-model (α for cure, γ for timing); users
    pass raw covariates only — no need to add a ones column.

    Note: the Weibull shape parameter is named ``shape`` rather than ``alpha`` to
    avoid collision with the cure sub-model's logistic intercept ``alpha``.
    """

    default_priors: ClassVar[PriorSpec] = {
        "alpha": (pm.Normal, {"mu": 0, "sigma": 1}),
        "beta_cure": (pm.Normal, {"mu": 0, "sigma": 3}),
        "gamma": (pm.Normal, {"mu": 0, "sigma": 1}),
        "delta": (pm.Normal, {"mu": 0, "sigma": 2}),
        "shape": (pm.Gamma, {"alpha": 5, "beta": 2}),
    }

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        self._n_features = X.shape[1]
        x_bar = X.mean(axis=0)

        with pm.Model() as model:
            # Covariates are centred for sampling; see _cure_sub_model for why.
            X_data = pm.Data("X", (X - x_bar).astype(float))
            t_data = pm.Data("t_obs", t.astype(float))
            event_data = pm.Data("event", event.astype(float))

            # Cure sub-model: π(x) = sigmoid(α + X·β_cure)
            logit_pi, beta_cure = _cure_sub_model(self, X_data, x_bar, X.shape[1])

            # Timing sub-model: Weibull with log-scale mean γ + X·δ
            log_lam, delta = _timing_linpred(self, X_data, x_bar, X.shape[1])
            shape = self._prior("shape")
            lam = pt.exp(log_lam)  # (n_obs,)

            # log S_u(t | x) = -(t / λ)^shape  (exact, no additional log needed)
            log_S = -((t_data / lam) ** shape)

            log_f = pm.logp(pm.Weibull.dist(alpha=shape, beta=lam), t_data)

            _register_mixture_likelihood(logit_pi, log_S, log_f, event_data)

        return model

    def _attach_log_likelihood(self) -> None:
        _lift_obs_loglik(self.idata)

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = self._check_n_features(X)
        times = np.asarray(times, dtype=float)

        assert self.idata is not None
        posterior = self.idata.posterior

        alpha_s = posterior["alpha"].values  # (chains, draws)
        beta_cure_s = posterior["beta_cure"].values  # (chains, draws, n_features)
        gamma_s = posterior["gamma"].values  # (chains, draws)
        delta_s = posterior["delta"].values  # (chains, draws, n_features)
        shape_s = posterior["shape"].values  # (chains, draws)

        n_chains, n_draws, n_features = beta_cure_s.shape
        n_samples = n_chains * n_draws

        alpha_flat = alpha_s.reshape(n_samples)  # (n_samples,)
        beta_cure_flat = beta_cure_s.reshape(n_samples, n_features)  # (n_samples, n_features)
        gamma_flat = gamma_s.reshape(n_samples)  # (n_samples,)
        delta_flat = delta_s.reshape(n_samples, n_features)  # (n_samples, n_features)
        shape_flat = shape_s.reshape(n_samples)  # (n_samples,)

        # pi: (n_samples, n_obs)
        pi = expit(alpha_flat[:, np.newaxis] + beta_cure_flat @ X.T)

        # lam: (n_samples, n_obs)
        lam = np.exp(gamma_flat[:, np.newaxis] + delta_flat @ X.T)

        # S_u: Weibull survival, shape (n_samples, n_obs, n_times)
        S_u = np.exp(
            -(
                (times[np.newaxis, np.newaxis, :] / lam[:, :, np.newaxis])
                ** shape_flat[:, np.newaxis, np.newaxis]
            )
        )

        # S_mix = π · S_u + (1 - π), broadcasting pi to (n_samples, n_obs, 1)
        return pi[:, :, np.newaxis] * S_u + (1.0 - pi[:, :, np.newaxis])

    def predict_cure_probability(
        self,
        X: np.ndarray,
        hdi_prob: float = 0.94,
    ) -> CurePrediction:
        """Posterior estimate of P(cured | x) = 1 - π(x) for each observation.

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        hdi_prob : credible interval width (default 0.94).

        Returns
        -------
        CurePrediction with arrays of shape (n_obs,).
        """
        self._check_fitted()
        X = self._check_n_features(X)

        assert self.idata is not None
        posterior = self.idata.posterior

        alpha_flat = posterior["alpha"].values.ravel()  # (n_samples,)
        beta_cure_flat = posterior["beta_cure"].values.reshape(
            -1, X.shape[1]
        )  # (n_samples, p)

        # cure_prob[s, i] = 1 - sigmoid(α_s + x_i · β_s) = P(cured | x_i, θ_s)
        cure_prob = 1.0 - expit(
            alpha_flat[:, np.newaxis] + beta_cure_flat @ X.T
        )  # (n_samples, n_obs)

        mean = cure_prob.mean(axis=0)
        hdi = np.stack(
            [az.hdi(cure_prob[:, i], hdi_prob=hdi_prob) for i in range(X.shape[0])],
            axis=0,
        )  # (n_obs, 2)

        return CurePrediction(mean=mean, hdi_lower=hdi[:, 0], hdi_upper=hdi[:, 1])

    def sample_predicted_event_times(
        self,
        X: np.ndarray,
        return_idata: bool = False,
        random_seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Draw posterior-predictive event times from the mixture cure model.

        Susceptible subjects (drawn from Bernoulli(π)) get a Weibull event time;
        non-susceptibles receive ``np.inf`` (they never experience the event).

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        return_idata : must be False; InferenceData is not available for numpy samples.
        random_seed : seed for the predictive draws; pass an int for reproducibility.

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs).
        """
        if return_idata:
            raise NotImplementedError(
                "return_idata=True is not supported for WeibullCureModel — "
                "samples are drawn in numpy and have no InferenceData wrapper."
            )
        rng = np.random.default_rng(random_seed)
        self._check_fitted()
        X = self._check_n_features(X)

        assert self.idata is not None
        posterior = self.idata.posterior

        alpha_s = posterior["alpha"].values
        beta_cure_s = posterior["beta_cure"].values
        gamma_s = posterior["gamma"].values
        delta_s = posterior["delta"].values
        shape_s = posterior["shape"].values

        n_chains, n_draws, n_features = beta_cure_s.shape
        n_samples = n_chains * n_draws

        alpha_flat = alpha_s.reshape(n_samples)
        beta_cure_flat = beta_cure_s.reshape(n_samples, n_features)
        gamma_flat = gamma_s.reshape(n_samples)
        delta_flat = delta_s.reshape(n_samples, n_features)
        shape_flat = shape_s.reshape(n_samples)

        pi = expit(
            alpha_flat[:, np.newaxis] + beta_cure_flat @ X.T
        )  # (n_samples, n_obs)
        lam = np.exp(
            gamma_flat[:, np.newaxis] + delta_flat @ X.T
        )  # (n_samples, n_obs)

        susceptible = rng.binomial(1, pi).astype(bool)  # (n_samples, n_obs)
        # rng.weibull draws from Weibull(shape, scale=1); multiply by lam for scale λ.
        # size=lam.shape is load-bearing: shape_flat[:, None] alone is (n_samples, 1),
        # so every observation would share one standard-Weibull draw per posterior
        # sample — marginals stay correct but the draws become perfectly rank-correlated
        # across individuals, corrupting any joint quantity.
        t_latent = lam * rng.weibull(
            shape_flat[:, np.newaxis], size=lam.shape
        )  # (n_samples, n_obs)

        return np.where(susceptible, t_latent, np.inf)


class LogLogisticCureModel(BaseSurvivalModel):
    """Mixture cure survival model with log-logistic timing distribution.

    Mixture survival function:
        S_mix(t | x) = π(x) · S_u(t | x) + (1 - π(x))

    where:
        π(x)       = sigmoid(α + X·β_cure)          — P(susceptible | x)
        S_u(t | x) = 1 / (1 + (t / λ(x))^shape)    — log-logistic survival among susceptibles
        λ(x)       = exp(γ + X·δ)                   — log-logistic scale (log link)

    Subjects in the "cured" fraction (probability 1 - π) never experience the event,
    so their mixture survival function asymptotes to 1 - π instead of zero.

    An intercept is included in each sub-model (α for cure, γ for timing); users
    pass raw covariates only — no need to add a ones column.

    Note: the log-logistic shape parameter is named ``shape`` rather than ``alpha`` to
    avoid collision with the cure sub-model's logistic intercept ``alpha``.
    """

    default_priors: ClassVar[PriorSpec] = {
        "alpha": (pm.Normal, {"mu": 0, "sigma": 1}),
        "beta_cure": (pm.Normal, {"mu": 0, "sigma": 3}),
        "gamma": (pm.Normal, {"mu": 0, "sigma": 1}),
        "delta": (pm.Normal, {"mu": 0, "sigma": 2}),
        "shape": (pm.Gamma, {"alpha": 5, "beta": 2}),
    }

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        self._n_features = X.shape[1]
        x_bar = X.mean(axis=0)

        with pm.Model() as model:
            # Covariates are centred for sampling; see _cure_sub_model for why.
            X_data = pm.Data("X", (X - x_bar).astype(float))
            t_data = pm.Data("t_obs", t.astype(float))
            event_data = pm.Data("event", event.astype(float))

            # Cure sub-model: π(x) = sigmoid(α + X·β_cure)
            logit_pi, beta_cure = _cure_sub_model(self, X_data, x_bar, X.shape[1])

            # Timing sub-model: log-logistic with log-scale mean γ + X·δ
            log_lam, delta = _timing_linpred(self, X_data, x_bar, X.shape[1])
            shape = self._prior("shape")
            lam = pt.exp(log_lam)  # (n_obs,)

            # log S_u(t | x) = -log(1 + (t/λ)^shape)
            log_S = -pt.log1p((t_data / lam) ** shape)

            # log f_u(t | x) = log(shape/λ) + (shape-1)·log(t/λ) + 2·log_S
            log_f = (
                pt.log(shape)
                - pt.log(lam)
                + (shape - 1) * (pt.log(t_data) - pt.log(lam))
                + 2 * log_S
            )

            _register_mixture_likelihood(logit_pi, log_S, log_f, event_data)

        return model

    def _attach_log_likelihood(self) -> None:
        _lift_obs_loglik(self.idata)

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = self._check_n_features(X)
        times = np.asarray(times, dtype=float)

        assert self.idata is not None
        posterior = self.idata.posterior

        alpha_s = posterior["alpha"].values  # (chains, draws)
        beta_cure_s = posterior["beta_cure"].values  # (chains, draws, n_features)
        gamma_s = posterior["gamma"].values  # (chains, draws)
        delta_s = posterior["delta"].values  # (chains, draws, n_features)
        shape_s = posterior["shape"].values  # (chains, draws)

        n_chains, n_draws, n_features = beta_cure_s.shape
        n_samples = n_chains * n_draws

        alpha_flat = alpha_s.reshape(n_samples)  # (n_samples,)
        beta_cure_flat = beta_cure_s.reshape(n_samples, n_features)  # (n_samples, n_features)
        gamma_flat = gamma_s.reshape(n_samples)  # (n_samples,)
        delta_flat = delta_s.reshape(n_samples, n_features)  # (n_samples, n_features)
        shape_flat = shape_s.reshape(n_samples)  # (n_samples,)

        # pi: (n_samples, n_obs)
        pi = expit(alpha_flat[:, np.newaxis] + beta_cure_flat @ X.T)

        # lam: (n_samples, n_obs)
        lam = np.exp(gamma_flat[:, np.newaxis] + delta_flat @ X.T)

        # S_u: log-logistic survival, shape (n_samples, n_obs, n_times)
        # S_u = 1 / (1 + (t/lam)^shape) = expit(shape * (log lam - log t))
        lam_exp = lam[:, :, np.newaxis]
        shape_exp = shape_flat[:, np.newaxis, np.newaxis]
        S_u = expit(shape_exp * (np.log(lam_exp) - np.log(times[np.newaxis, np.newaxis, :])))

        # S_mix = π · S_u + (1 - π), broadcasting pi to (n_samples, n_obs, 1)
        return pi[:, :, np.newaxis] * S_u + (1.0 - pi[:, :, np.newaxis])

    def predict_cure_probability(
        self,
        X: np.ndarray,
        hdi_prob: float = 0.94,
    ) -> CurePrediction:
        """Posterior estimate of P(cured | x) = 1 - π(x) for each observation.

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        hdi_prob : credible interval width (default 0.94).

        Returns
        -------
        CurePrediction with arrays of shape (n_obs,).
        """
        self._check_fitted()
        X = self._check_n_features(X)

        assert self.idata is not None
        posterior = self.idata.posterior

        alpha_flat = posterior["alpha"].values.ravel()  # (n_samples,)
        beta_cure_flat = posterior["beta_cure"].values.reshape(
            -1, X.shape[1]
        )  # (n_samples, p)

        # cure_prob[s, i] = 1 - sigmoid(α_s + x_i · β_s) = P(cured | x_i, θ_s)
        cure_prob = 1.0 - expit(
            alpha_flat[:, np.newaxis] + beta_cure_flat @ X.T
        )  # (n_samples, n_obs)

        mean = cure_prob.mean(axis=0)
        hdi = np.stack(
            [az.hdi(cure_prob[:, i], hdi_prob=hdi_prob) for i in range(X.shape[0])],
            axis=0,
        )  # (n_obs, 2)

        return CurePrediction(mean=mean, hdi_lower=hdi[:, 0], hdi_upper=hdi[:, 1])

    def sample_predicted_event_times(
        self,
        X: np.ndarray,
        return_idata: bool = False,
        random_seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Draw posterior-predictive event times from the mixture cure model.

        Susceptible subjects (drawn from Bernoulli(π)) get a log-logistic event time;
        non-susceptibles receive ``np.inf`` (they never experience the event).

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        return_idata : must be False; InferenceData is not available for numpy samples.
        random_seed : seed for the predictive draws; pass an int for reproducibility.

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs).
        """
        if return_idata:
            raise NotImplementedError(
                "return_idata=True is not supported for LogLogisticCureModel — "
                "samples are drawn in numpy and have no InferenceData wrapper."
            )
        rng = np.random.default_rng(random_seed)
        self._check_fitted()
        X = self._check_n_features(X)

        assert self.idata is not None
        posterior = self.idata.posterior

        alpha_s = posterior["alpha"].values
        beta_cure_s = posterior["beta_cure"].values
        gamma_s = posterior["gamma"].values
        delta_s = posterior["delta"].values
        shape_s = posterior["shape"].values

        n_chains, n_draws, n_features = beta_cure_s.shape
        n_samples = n_chains * n_draws

        alpha_flat = alpha_s.reshape(n_samples)
        beta_cure_flat = beta_cure_s.reshape(n_samples, n_features)
        gamma_flat = gamma_s.reshape(n_samples)
        delta_flat = delta_s.reshape(n_samples, n_features)
        shape_flat = shape_s.reshape(n_samples)

        pi = expit(
            alpha_flat[:, np.newaxis] + beta_cure_flat @ X.T
        )  # (n_samples, n_obs)
        lam = np.exp(
            gamma_flat[:, np.newaxis] + delta_flat @ X.T
        )  # (n_samples, n_obs)

        susceptible = rng.binomial(1, pi).astype(bool)  # (n_samples, n_obs)
        # Log-logistic quantile: T = λ · (U / (1-U))^(1/shape) for U ~ Uniform(0,1)
        U = rng.uniform(size=pi.shape)
        t_latent = lam * (U / (1.0 - U)) ** (1.0 / shape_flat[:, np.newaxis])

        return np.where(susceptible, t_latent, np.inf)
