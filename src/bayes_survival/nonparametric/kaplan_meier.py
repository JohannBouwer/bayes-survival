"""Bayesian Kaplan-Meier estimator via Beta-Binomial conjugate model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import arviz as az
import numpy as np
from scipy.stats import beta as BetaDist

from bayes_survival.survival_models.base import SurvivalPrediction

# Prior spec for conjugate Beta model: name -> (alpha, beta)
BetaPriorSpec = dict[str, tuple[float, float]]


@dataclass
class _RiskTable:
    times: np.ndarray      # sorted unique event times, shape (n_event_times,)
    n_at_risk: np.ndarray  # number at risk at each event time, shape (n_event_times,)
    n_events: np.ndarray   # number of events at each event time, shape (n_event_times,)


class KaplanMeierModel:
    """Bayesian Kaplan-Meier estimator using the Beta-Binomial conjugate model.

    At each distinct event time t_j, the conditional hazard
    h_j = P(event at t_j | survived to t_j) is assigned an independent Beta prior.
    The Beta-Binomial conjugacy yields an exact closed-form posterior:

        Prior:      h_j ~ Beta(alpha, beta)
        Likelihood: d_j | h_j ~ Binomial(n_j, h_j)
        Posterior:  h_j | data ~ Beta(alpha + d_j, beta + n_j - d_j)

    The survival function is estimated as the product of conditional survival
    probabilities over all event times up to t:

        S(t) = prod_{t_j <= t} (1 - h_j)

    The posterior mean has a closed form:

        E[S(t)] = prod_{t_j <= t} beta_post_j / (alpha_post_j + beta_post_j)

    because the h_j are posterior-independent across event times.

    Parameters
    ----------
    priors : dict[str, tuple[float, float]], optional
        Override default priors. The only valid key is ``"h"``, a ``(alpha, beta)``
        tuple specifying the Beta prior placed on each conditional hazard.
        Both values must be strictly positive.

        Common choices:

        - ``(1.0, 1.0)`` — Uniform / Bayes-Laplace (default)
        - ``(0.5, 0.5)`` — Jeffreys prior

    Examples
    --------
    >>> import numpy as np
    >>> from bayes_survival.nonparametric import KaplanMeierModel
    >>> km = KaplanMeierModel()
    >>> km.fit(t, event)
    >>> pred = km.predict_survival_function(times=np.linspace(0, 20, 100))
    >>> pred.mean.shape
    (1, 100)
    """

    default_priors: ClassVar[BetaPriorSpec] = {
        "h": (1.0, 1.0),  # Beta(1, 1) = Uniform prior on each conditional hazard
    }

    def __init__(self, priors: BetaPriorSpec | None = None) -> None:
        self.priors: BetaPriorSpec = dict(type(self).default_priors)
        if priors is not None:
            self._validate_and_merge_priors(priors)
        self._risk_table: _RiskTable | None = None
        self._alpha_post: np.ndarray | None = None
        self._beta_post: np.ndarray | None = None

    def _validate_and_merge_priors(self, priors: BetaPriorSpec) -> None:
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
                    f"Beta prior parameters for '{name}' must be strictly positive,"
                    f" got ({a}, {b})"
                )
            self.priors[name] = spec

    def fit(self, t: np.ndarray, event: np.ndarray) -> KaplanMeierModel:
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
        self._beta_post = beta_prior + (n - d)

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

    def sample_posterior_survival(
        self,
        times: np.ndarray,
        n_samples: int = 2000,
    ) -> np.ndarray:
        """Draw survival curve samples from the posterior.

        Parameters
        ----------
        times : array-like of shape (n_times,)
            Time points at which to evaluate the survival function.
        n_samples : int
            Number of posterior samples to draw.

        Returns
        -------
        np.ndarray of shape (n_samples, n_times)
            Each row is a survival curve drawn from the posterior.
        """
        self._check_fitted()
        assert self._alpha_post is not None and self._beta_post is not None
        times = np.asarray(times, dtype=float)

        # h_samples[s, j] ~ Beta(alpha_post_j, beta_post_j)
        # shape: (n_samples, n_event_times)
        h_samples = np.column_stack(
            [
                BetaDist.rvs(a, b, size=n_samples)
                for a, b in zip(self._alpha_post, self._beta_post)
            ]
        )

        event_times = self._risk_table.times  # type: ignore[union-attr]
        S_samples = np.ones((n_samples, len(times)))
        for k, t_query in enumerate(times):
            mask = event_times <= t_query
            if mask.any():
                S_samples[:, k] = np.prod(1.0 - h_samples[:, mask], axis=1)

        return S_samples

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
            Time points at which to evaluate the survival function.
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
    def posterior_mean_survival(self) -> tuple[np.ndarray, np.ndarray]:
        """Analytically computed posterior mean survival at event times.

        Exploits the closed-form result:

            E[S(t)] = prod_{t_j <= t} E[1 - h_j]
                    = prod_{t_j <= t} beta_post_j / (alpha_post_j + beta_post_j)

        which follows from the posterior independence of h_j across event times.

        Returns
        -------
        times : np.ndarray of shape (n_event_times,)
            Sorted unique event times.
        S_mean : np.ndarray of shape (n_event_times,)
            Posterior mean survival at each event time (cumulative product).
        """
        self._check_fitted()
        assert self._alpha_post is not None and self._beta_post is not None
        survival_factor = self._beta_post / (self._alpha_post + self._beta_post)
        S_mean = np.cumprod(survival_factor)
        return self._risk_table.times, S_mean  # type: ignore[union-attr]

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
