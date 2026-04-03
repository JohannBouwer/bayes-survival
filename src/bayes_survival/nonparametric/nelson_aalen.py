"""Bayesian Nelson-Aalen estimator via Gamma-Poisson conjugate model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import arviz as az
import numpy as np
from scipy.stats import gamma as GammaDist

from bayes_survival.survival_models.base import SurvivalPrediction

# Prior spec for conjugate Gamma model: name -> (alpha, beta)  [rate parameterisation]
GammaPriorSpec = dict[str, tuple[float, float]]


@dataclass
class _RiskTable:
    times: np.ndarray      # sorted unique event times, shape (n_event_times,)
    n_at_risk: np.ndarray  # number at risk at each event time, shape (n_event_times,)
    n_events: np.ndarray   # number of events at each event time, shape (n_event_times,)


class NelsonAalenModel:
    """Bayesian Nelson-Aalen cumulative hazard estimator using the Gamma-Poisson conjugate model.

    At each distinct event time t_j, the hazard increment
    λ_j (expected events per unit at-risk) is assigned an independent Gamma prior.
    The Gamma-Poisson conjugacy yields an exact closed-form posterior:

        Prior:      λ_j ~ Gamma(alpha, beta)          [rate parameterisation]
        Likelihood: d_j | λ_j ~ Poisson(n_j * λ_j)
        Posterior:  λ_j | data ~ Gamma(alpha + d_j, beta + n_j)

    The cumulative hazard is estimated as the sum of hazard increments up to t:

        H(t) = sum_{t_j <= t} λ_j

    The survival function is recovered via the relation:

        S(t) = exp(-H(t))

    The posterior mean of H has a closed form:

        E[H(t)] = sum_{t_j <= t} (alpha + d_j) / (beta + n_j)

    because the λ_j are posterior-independent across event times.

    Parameters
    ----------
    priors : dict[str, tuple[float, float]], optional
        Override default priors. The only valid key is ``"h"``, an ``(alpha, beta)``
        tuple specifying the Gamma prior placed on each hazard increment.
        Both values must be strictly positive.

        Common choices:

        - ``(0.1, 0.1)`` — vague / near-Jeffreys (default)
        - ``(1.0, 1.0)`` — Exponential(1) prior

    Examples
    --------
    >>> import numpy as np
    >>> from bayes_survival.nonparametric import NelsonAalenModel
    >>> na = NelsonAalenModel()
    >>> na.fit(t, event)
    >>> pred = na.predict_cumulative_hazard(times=np.linspace(0, 20, 100))
    >>> pred.mean.shape
    (1, 100)
    """

    default_priors: ClassVar[GammaPriorSpec] = {
        "h": (0.1, 0.1),  # Gamma(0.1, 0.1) — vague prior on each hazard increment
    }

    def __init__(self, priors: GammaPriorSpec | None = None) -> None:
        self.priors: GammaPriorSpec = dict(type(self).default_priors)
        if priors is not None:
            self._validate_and_merge_priors(priors)
        self._risk_table: _RiskTable | None = None
        self._alpha_post: np.ndarray | None = None
        self._beta_post: np.ndarray | None = None

    def _validate_and_merge_priors(self, priors: GammaPriorSpec) -> None:
        valid_keys = set(type(self).default_priors)
        unknown = set(priors) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown prior keys: {unknown}. Valid keys for this model: {valid_keys}"
            )
        for name, spec in priors.items():
            if not (isinstance(spec, tuple) and len(spec) == 2):
                raise TypeError(
                    f"Prior '{name}' must be an (alpha, beta) tuple, got {type(spec)}"
                )
            a, b = spec
            if a <= 0 or b <= 0:
                raise ValueError(
                    f"Gamma prior parameters for '{name}' must be strictly positive,"
                    f" got ({a}, {b})"
                )
            self.priors[name] = spec

    def fit(self, t: np.ndarray, event: np.ndarray) -> NelsonAalenModel:
        """Fit the model to observed survival data.

        Parameters
        ----------
        t : array-like of shape (n_obs,)
            Observed times — either the exact event time or the last follow-up
            time for censored observations.
        event : array-like of shape (n_obs,)
            Event indicator: ``1`` = event observed, ``0`` = right-censored.

        Returns
        -------
        self
        """
        t = np.asarray(t, dtype=float)
        event = np.asarray(event, dtype=float)

        self._risk_table = self._compute_risk_table(t, event)

        alpha_prior, beta_prior = self.priors["h"]
        d = self._risk_table.n_events
        n = self._risk_table.n_at_risk

        self._alpha_post = alpha_prior + d
        self._beta_post = beta_prior + n

        return self

    @staticmethod
    def _compute_risk_table(t: np.ndarray, event: np.ndarray) -> _RiskTable:
        event_times = np.sort(np.unique(t[event == 1]))
        n_at_risk = np.array([np.sum(t >= tj) for tj in event_times], dtype=int)
        n_events = np.array(
            [np.sum((t == tj) & (event == 1)) for tj in event_times], dtype=int
        )
        return _RiskTable(times=event_times, n_at_risk=n_at_risk, n_events=n_events)

    def _check_fitted(self) -> None:
        if self._risk_table is None:
            raise RuntimeError("Model has not been fitted yet — call fit() first.")

    def sample_posterior_cumulative_hazard(
        self,
        times: np.ndarray,
        n_samples: int = 2000,
    ) -> np.ndarray:
        """Draw cumulative hazard curve samples from the posterior.

        Parameters
        ----------
        times : array-like of shape (n_times,)
            Time points at which to evaluate H(t).
        n_samples : int
            Number of posterior samples to draw.

        Returns
        -------
        np.ndarray of shape (n_samples, n_times)
            Each row is a cumulative hazard curve drawn from the posterior.
        """
        self._check_fitted()
        assert self._alpha_post is not None and self._beta_post is not None
        times = np.asarray(times, dtype=float)

        # lambda_samples[s, j] ~ Gamma(alpha_post_j, scale=1/beta_post_j)
        # shape: (n_samples, n_event_times)
        lambda_samples = GammaDist.rvs(
            a=self._alpha_post,
            scale=1.0 / self._beta_post,
            size=(n_samples, len(self._risk_table.times)),
        )

        event_times = self._risk_table.times  # type: ignore[union-attr]
        H_samples = np.zeros((n_samples, len(times)))
        for k, t_query in enumerate(times):
            mask = event_times <= t_query
            if mask.any():
                H_samples[:, k] = lambda_samples[:, mask].sum(axis=1)

        return H_samples

    def sample_posterior_survival(
        self,
        times: np.ndarray,
        n_samples: int = 2000,
    ) -> np.ndarray:
        """Draw survival curve samples from the posterior via S(t) = exp(-H(t)).

        Parameters
        ----------
        times : array-like of shape (n_times,)
            Time points at which to evaluate S(t).
        n_samples : int
            Number of posterior samples to draw.

        Returns
        -------
        np.ndarray of shape (n_samples, n_times)
            Each row is a survival curve drawn from the posterior.
        """
        H_samples = self.sample_posterior_cumulative_hazard(times, n_samples=n_samples)
        return np.exp(-H_samples)

    def predict_cumulative_hazard(
        self,
        times: np.ndarray,
        hdi_prob: float = 0.94,
        n_samples: int = 2000,
    ) -> SurvivalPrediction:
        """Posterior cumulative hazard estimate at the given times.

        Parameters
        ----------
        times : array-like of shape (n_times,)
            Time points at which to evaluate H(t).
        hdi_prob : float
            Probability mass within the highest-density interval.
        n_samples : int
            Number of posterior samples used for estimation.

        Returns
        -------
        SurvivalPrediction
            ``mean``, ``hdi_lower``, ``hdi_upper`` each have shape ``(1, n_times)``.
            ``times`` has shape ``(n_times,)``.
        """
        self._check_fitted()
        times = np.asarray(times, dtype=float)
        H_samples = self.sample_posterior_cumulative_hazard(times, n_samples=n_samples)
        # Expand to (n_samples, 1, n_times) — treat as single observation (no covariates)
        return self._aggregate(H_samples[:, np.newaxis, :], times, hdi_prob)

    def predict_survival_function(
        self,
        times: np.ndarray,
        hdi_prob: float = 0.94,
        n_samples: int = 2000,
    ) -> SurvivalPrediction:
        """Posterior survival function estimate at the given times.

        Parameters
        ----------
        times : array-like of shape (n_times,)
            Time points at which to evaluate S(t).
        hdi_prob : float
            Probability mass within the highest-density interval.
        n_samples : int
            Number of posterior samples used for estimation.

        Returns
        -------
        SurvivalPrediction
            ``mean``, ``hdi_lower``, ``hdi_upper`` each have shape ``(1, n_times)``.
            ``times`` has shape ``(n_times,)``.
        """
        self._check_fitted()
        times = np.asarray(times, dtype=float)
        S_samples = self.sample_posterior_survival(times, n_samples=n_samples)
        # Expand to (n_samples, 1, n_times) — treat as single observation (no covariates)
        return self._aggregate(S_samples[:, np.newaxis, :], times, hdi_prob)

    def survival_probability(
        self,
        t: float,
        hdi_prob: float = 0.94,
        n_samples: int = 2000,
    ) -> SurvivalPrediction:
        """Posterior probability of surviving past time t.

        Parameters
        ----------
        t : float
            Query time.
        hdi_prob : float
            Probability mass within the highest-density interval.
        n_samples : int
            Number of posterior samples.

        Returns
        -------
        SurvivalPrediction with all arrays of shape (1, 1).
        """
        return self.predict_survival_function(
            times=np.array([t]),
            hdi_prob=hdi_prob,
            n_samples=n_samples,
        )

    @property
    def posterior_mean_cumulative_hazard(self) -> tuple[np.ndarray, np.ndarray]:
        """Analytically computed posterior mean cumulative hazard at event times.

        Exploits the closed-form result:

            E[H(t)] = sum_{t_j <= t} E[lambda_j]
                    = sum_{t_j <= t} (alpha + d_j) / (beta + n_j)

        which follows from the posterior independence of lambda_j across event times.

        Returns
        -------
        times : np.ndarray of shape (n_event_times,)
            Sorted unique event times.
        H_mean : np.ndarray of shape (n_event_times,)
            Posterior mean cumulative hazard at each event time (cumulative sum).
        """
        self._check_fitted()
        assert self._alpha_post is not None and self._beta_post is not None
        hazard_increments = self._alpha_post / self._beta_post
        H_mean = np.cumsum(hazard_increments)
        return self._risk_table.times, H_mean  # type: ignore[union-attr]

    @property
    def posterior_mean_survival(self) -> tuple[np.ndarray, np.ndarray]:
        """Analytically computed posterior mean survival at event times via exp(-H).

        Returns
        -------
        times : np.ndarray of shape (n_event_times,)
            Sorted unique event times.
        S_mean : np.ndarray of shape (n_event_times,)
            exp(-E[H(t)]) at each event time.
        """
        times, H_mean = self.posterior_mean_cumulative_hazard
        return times, np.exp(-H_mean)

    @staticmethod
    def _aggregate(
        samples: np.ndarray,
        times: np.ndarray,
        hdi_prob: float,
    ) -> SurvivalPrediction:
        """Compute posterior mean and HDI from samples.

        Parameters
        ----------
        samples : np.ndarray of shape (n_samples, n_obs, n_times)
        times : np.ndarray of shape (n_times,)
        hdi_prob : float

        Returns
        -------
        SurvivalPrediction with mean/hdi_lower/hdi_upper of shape (n_obs, n_times).
        """
        mean = samples.mean(axis=0)  # (n_obs, n_times)
        hdi = np.stack(
            [
                az.hdi(samples[:, i, :], hdi_prob=hdi_prob)
                for i in range(samples.shape[1])
            ],
            axis=0,
        )  # (n_obs, n_times, 2)
        return SurvivalPrediction(
            times=times,
            mean=mean,
            hdi_lower=hdi[..., 0],
            hdi_upper=hdi[..., 1],
        )
