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


class LogNormalCureModel(BaseSurvivalModel):
    """Mixture cure survival model with log-normal timing distribution.

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

        with pm.Model() as model:
            X_data = pm.Data("X", X.astype(float))
            t_data = pm.Data("t_obs", t.astype(float))
            event_data = pm.Data("event", event.astype(float))

            # Cure sub-model: π(x) = sigmoid(\alpha + X·β_cure)
            alpha = self._prior("alpha")
            beta_cure = self._prior("beta_cure", shape=X.shape[1])
            pi = pm.Deterministic(
                "pi", pm.math.sigmoid(alpha + pt.dot(X_data, beta_cure))
            )

            # Timing sub-model: log-normal with mean \gamma + X·δ on the log scale
            gamma = self._prior("gamma")
            delta = self._prior("delta", shape=X.shape[1])
            sigma = self._prior("sigma")
            mu = gamma + pt.dot(X_data, delta)

            # log S_u(t | x) = log(0.5 · erfc((log t - μ) / (\sigma√2)))
            log_S = pt.log(
                0.5 * pt.erfc((pt.log(t_data) - mu) / (sigma * pt.sqrt(2.0)))
            )

            log_f = pm.logp(pm.LogNormal.dist(mu=mu, sigma=sigma), t_data)

            log_lik_event = pt.log(pi) + log_f
            log_lik_censored = pt.log(pi * pt.exp(log_S) + (1.0 - pi))

            pm.Potential(
                "obs",
                pt.switch(pt.eq(event_data, 1.0), log_lik_event, log_lik_censored),
            )

        return model

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = np.atleast_2d(X)
        times = np.asarray(times, dtype=float)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

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
        X = np.atleast_2d(X)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

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
        **kwargs,
    ) -> np.ndarray:
        """Draw posterior-predictive event times from the mixture cure model.

        Susceptible subjects (drawn from Bernoulli(π)) get a log-normal event time;
        non-susceptibles receive ``np.inf`` (they never experience the event).

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        return_idata : must be False; InferenceData is not available for numpy samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs).
        """
        if return_idata:
            raise NotImplementedError(
                "return_idata=True is not supported for LogNormalCureModel — "
                "samples are drawn in numpy and have no InferenceData wrapper."
            )
        self._check_fitted()
        X = np.atleast_2d(X)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

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

        susceptible = np.random.binomial(1, pi).astype(bool)  # (n_samples, n_obs)
        t_latent = np.random.lognormal(
            mu, sigma_flat[:, np.newaxis]
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

        with pm.Model() as model:
            X_data = pm.Data("X", X.astype(float))
            t_data = pm.Data("t_obs", t.astype(float))
            event_data = pm.Data("event", event.astype(float))

            # Cure sub-model: π(x) = sigmoid(α + X·β_cure)
            alpha = self._prior("alpha")
            beta_cure = self._prior("beta_cure", shape=X.shape[1])
            pi = pm.Deterministic(
                "pi", pm.math.sigmoid(alpha + pt.dot(X_data, beta_cure))
            )

            # Timing sub-model: Weibull with log-scale mean γ + X·δ
            gamma = self._prior("gamma")
            delta = self._prior("delta", shape=X.shape[1])
            shape = self._prior("shape")
            lam = pt.exp(gamma + pt.dot(X_data, delta))  # (n_obs,)

            # log S_u(t | x) = -(t / λ)^shape  (exact, no additional log needed)
            log_S = -((t_data / lam) ** shape)

            log_f = pm.logp(pm.Weibull.dist(alpha=shape, beta=lam), t_data)

            log_lik_event = pt.log(pi) + log_f
            log_lik_censored = pt.log(pi * pt.exp(log_S) + (1.0 - pi))

            pm.Potential(
                "obs",
                pt.switch(pt.eq(event_data, 1.0), log_lik_event, log_lik_censored),
            )

        return model

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = np.atleast_2d(X)
        times = np.asarray(times, dtype=float)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

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
        X = np.atleast_2d(X)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

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
        **kwargs,
    ) -> np.ndarray:
        """Draw posterior-predictive event times from the mixture cure model.

        Susceptible subjects (drawn from Bernoulli(π)) get a Weibull event time;
        non-susceptibles receive ``np.inf`` (they never experience the event).

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        return_idata : must be False; InferenceData is not available for numpy samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs).
        """
        if return_idata:
            raise NotImplementedError(
                "return_idata=True is not supported for WeibullCureModel — "
                "samples are drawn in numpy and have no InferenceData wrapper."
            )
        self._check_fitted()
        X = np.atleast_2d(X)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

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

        susceptible = np.random.binomial(1, pi).astype(bool)  # (n_samples, n_obs)
        # np.random.weibull draws from Weibull(shape, scale=1); multiply by lam for scale λ
        t_latent = lam * np.random.weibull(
            shape_flat[:, np.newaxis]
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

        with pm.Model() as model:
            X_data = pm.Data("X", X.astype(float))
            t_data = pm.Data("t_obs", t.astype(float))
            event_data = pm.Data("event", event.astype(float))

            # Cure sub-model: π(x) = sigmoid(α + X·β_cure)
            alpha = self._prior("alpha")
            beta_cure = self._prior("beta_cure", shape=X.shape[1])
            pi = pm.Deterministic(
                "pi", pm.math.sigmoid(alpha + pt.dot(X_data, beta_cure))
            )

            # Timing sub-model: log-logistic with log-scale mean γ + X·δ
            gamma = self._prior("gamma")
            delta = self._prior("delta", shape=X.shape[1])
            shape = self._prior("shape")
            lam = pt.exp(gamma + pt.dot(X_data, delta))  # (n_obs,)

            # log S_u(t | x) = -log(1 + (t/λ)^shape)
            log_S = -pt.log1p((t_data / lam) ** shape)

            # log f_u(t | x) = log(shape/λ) + (shape-1)·log(t/λ) + 2·log_S
            log_f = (
                pt.log(shape)
                - pt.log(lam)
                + (shape - 1) * (pt.log(t_data) - pt.log(lam))
                + 2 * log_S
            )

            log_lik_event = pt.log(pi) + log_f
            log_lik_censored = pt.log(pi * pt.exp(log_S) + (1.0 - pi))

            pm.Potential(
                "obs",
                pt.switch(pt.eq(event_data, 1.0), log_lik_event, log_lik_censored),
            )

        return model

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = np.atleast_2d(X)
        times = np.asarray(times, dtype=float)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

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
        X = np.atleast_2d(X)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

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
        **kwargs,
    ) -> np.ndarray:
        """Draw posterior-predictive event times from the mixture cure model.

        Susceptible subjects (drawn from Bernoulli(π)) get a log-logistic event time;
        non-susceptibles receive ``np.inf`` (they never experience the event).

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        return_idata : must be False; InferenceData is not available for numpy samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs).
        """
        if return_idata:
            raise NotImplementedError(
                "return_idata=True is not supported for LogLogisticCureModel — "
                "samples are drawn in numpy and have no InferenceData wrapper."
            )
        self._check_fitted()
        X = np.atleast_2d(X)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

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

        susceptible = np.random.binomial(1, pi).astype(bool)  # (n_samples, n_obs)
        # Log-logistic quantile: T = λ · (U / (1-U))^(1/shape) for U ~ Uniform(0,1)
        U = np.random.uniform(size=pi.shape)
        t_latent = lam * (U / (1.0 - U)) ** (1.0 / shape_flat[:, np.newaxis])

        return np.where(susceptible, t_latent, np.inf)
