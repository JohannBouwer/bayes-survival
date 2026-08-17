from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pymc as pm
import arviz as az

from bayes_survival._validation import validate_survival_inputs


PriorSpec = dict[str, tuple[type, dict]]


@dataclass
class HierarchySpec:
    """Specification for a group of covariates that share a hyper-prior.

    Covariates in the group are drawn from Normal(mu_group, sigma_group), where
    mu_group and sigma_group are themselves given priors (hyper-priors), enabling
    partial pooling across the group.

    Parameters
    ----------
    name:
        Identifier used as a prefix for PyMC variable names
        (``mu_{name}``, ``sigma_{name}``, ``beta_{name}``).
    covariate_names:
        Column names (matching the DataFrame passed to ``fit``) of the
        covariates that belong to this group.
    mu_prior:
        ``(DistributionClass, kwargs)`` tuple for the hyper-prior on the group mean.
        Defaults to ``Normal(mu=0, sigma=1)``.
    sigma_prior:
        ``(DistributionClass, kwargs)`` tuple for the hyper-prior on the group
        standard deviation.  Defaults to ``HalfNormal(sigma=1)``.
    """

    name: str
    covariate_names: list[str]
    mu_prior: tuple[type, dict] = field(
        default_factory=lambda: (pm.Normal, {"mu": 0, "sigma": 1})
    )
    sigma_prior: tuple[type, dict] = field(
        default_factory=lambda: (pm.HalfNormal, {"sigma": 1})
    )
    centered: bool = True


@dataclass
class SurvivalPrediction:
    times: np.ndarray  # (n_times,)
    mean: np.ndarray  # (n_obs, n_times)
    hdi_lower: np.ndarray  # (n_obs, n_times)
    hdi_upper: np.ndarray  # (n_obs, n_times)


class BaseSurvivalModel(ABC):
    """Abstract class define survival model interface"""

    default_priors: ClassVar[PriorSpec] = {}

    def __init__(self, priors: PriorSpec | None = None):
        self.priors: PriorSpec = dict(type(self).default_priors)
        if priors is not None:
            self._validate_and_merge_priors(priors)
        self.model: pm.Model | None = None
        self.idata: az.InferenceData | None = None
        self._n_features: int | None = None

    def _validate_and_merge_priors(self, priors: PriorSpec) -> None:
        """Validate user-supplied priors and merge them into self.priors."""
        valid_keys = set(type(self).default_priors)
        unknown = set(priors) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown prior keys: {unknown}. Valid keys for this model: {valid_keys}"
            )
        for name, spec in priors.items():
            if not (isinstance(spec, tuple) and len(spec) == 2):
                raise TypeError(
                    f"Prior '{name}' must be a (dist_class, kwargs) tuple, got {type(spec)}"
                )
            dist_cls, kwargs = spec
            if not (
                isinstance(dist_cls, type) and issubclass(dist_cls, pm.Distribution)
            ):
                raise TypeError(
                    f"Prior '{name}': first element must be a PyMC Distribution subclass"
                )
        self.priors.update(priors)

    def _prior(self, name: str, **extra_kwargs) -> pm.Distribution:
        """Instantiate a named prior inside the active model context.

        extra_kwargs are merged into the registered kwargs, allowing data-dependent
        parameters (e.g. shape=n_coef) to be injected at build time.
        """
        dist_cls, kwargs = self.priors[name]
        return dist_cls(name, **{**kwargs, **extra_kwargs})

    @staticmethod
    def _augment_X(X: np.ndarray) -> np.ndarray:
        """Prepend a column of ones (intercept) to X. Returns shape (n_obs, n_features + 1)."""
        return np.column_stack([np.ones(X.shape[0]), X])

    def _check_n_features(self, X: np.ndarray) -> np.ndarray:
        """Validate a prediction-time design matrix against the fitted one."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")
        return X

    @staticmethod
    def _flatten_ppc(idata_ppc: az.InferenceData) -> np.ndarray:
        """Flatten posterior-predictive "obs" draws to (n_chains * n_draws, n_obs)."""
        obs = idata_ppc.posterior_predictive["obs"].values  # (chains, draws, n_obs)
        n_chains, n_draws, n_obs = obs.shape
        return obs.reshape(n_chains * n_draws, n_obs)

    @abstractmethod
    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        """Construct and return the PyMC model without sampling.

        Subclasses that support sample_predicted_event_times() must register
        pm.Data containers named "X_aug", "t_obs", and "upper" in the model graph.
        Override sample_predicted_event_times() if a different structure is used.
        """
        ...

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
        draws: int = 1000,
        tune: int = 1000,
        nuts_sampler: str = "nutpie",
        log_likelihood: bool = True,
        **sample_kwargs,
    ) -> BaseSurvivalModel:
        """Fit the model by MCMC.

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        t : (n_obs,) — observed times, strictly positive.
        event : (n_obs,) — 1 = event, 0 = right-censored.
        log_likelihood : if True (default) attach a pointwise ``log_likelihood``
            group to ``self.idata`` so ``az.loo`` / ``az.compare`` can be used.
        **sample_kwargs : forwarded to ``pm.sample``.
        """
        X, t, event = validate_survival_inputs(
            t, event, X, model_name=type(self).__name__
        )
        self.model = self.build_model(X, t, event)
        with self.model:
            self.idata = pm.sample(draws=draws, tune=tune, nuts_sampler=nuts_sampler, **sample_kwargs)
            if log_likelihood:
                self._attach_log_likelihood()
        return self

    def _attach_log_likelihood(self) -> None:
        """Add a pointwise ``log_likelihood`` group to ``self.idata``.

        Must run inside ``fit`` — ``pm.compute_log_likelihood`` re-evaluates the
        graph against the model's *current* ``pm.Data`` contents, and
        ``sample_predicted_event_times`` overwrites those via ``pm.set_data``.

        Note this cannot be delegated to ``idata_kwargs={"log_likelihood": True}``:
        that flag is silently ignored by the external-sampler path, and this
        library defaults to ``nuts_sampler="nutpie"``.
        """
        assert self.model is not None and self.idata is not None
        pm.compute_log_likelihood(self.idata, model=self.model, progressbar=False)

    @abstractmethod
    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """
        Return raw posterior survival samples.
        Shape: (n_samples, n_obs, n_times), values in [0, 1].
        """
        ...

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: np.ndarray,
        hdi_prob: float = 0.94,
    ) -> SurvivalPrediction:
        self._check_fitted()
        samples = self._predict_survival_samples(X, times)
        return self._aggregate(samples, times, hdi_prob)

    def survival_probability(
        self,
        X: np.ndarray,
        t: float,
        hdi_prob: float = 0.94,
    ) -> SurvivalPrediction:
        """S(t | x): probability of surviving past time t."""
        return self.predict_survival_function(X, np.array([t]), hdi_prob=hdi_prob)

    def conditional_event_probability(
        self,
        X: np.ndarray,
        t: float,
        T: float,
        hdi_prob: float = 0.94,
    ) -> SurvivalPrediction:
        """
        P(event before T | survived to t, x) = 1 - S(T|x) / S(t|x).

        Answers: given an individual has survived to time t, what is the
        probability they experience the event before time T?
        Ratio is computed on raw samples before aggregation.
        """
        if T <= t:
            raise ValueError(f"T ({T}) must be greater than t ({t})")
        self._check_fitted()

        samples = self._predict_survival_samples(X, np.array([t, T]))
        # samples: (n_samples, n_obs, 2) — index 0=S(t), 1=S(T)
        s_t = samples[:, :, 0]
        s_T = samples[:, :, 1]

        # Clamp denominator to avoid division near zero (cured/long-tail cases)
        ratio = s_T / np.clip(s_t, a_min=1e-8, a_max=None)
        cond_prob_samples = 1.0 - ratio  # (n_samples, n_obs)

        return self._aggregate(
            cond_prob_samples[:, :, np.newaxis], np.array([T]), hdi_prob
        )

    @staticmethod
    def _aggregate(
        samples: np.ndarray,
        times: np.ndarray,
        hdi_prob: float,
    ) -> SurvivalPrediction:
        mean = samples.mean(axis=0)
        # az.hdi interprets 3D arrays as (chains, draws, vars), which would pool
        # across observations and return the wrong shape. Compute per observation.
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

    def sample_predicted_event_times(
        self,
        X: np.ndarray,
        return_idata: bool = False,
        random_seed: int | None = None,
        **sample_kwargs,
    ) -> np.ndarray | az.InferenceData:
        """Draw posterior-predictive event times for new observations.

        Sets upper=inf for every row so pm.Censored returns uncensored draws
        from the latent event-time distribution.

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        return_idata : if True, return the raw az.InferenceData object.
        random_seed : seed for the predictive draws; pass an int for reproducibility.
        **sample_kwargs : forwarded to pm.sample_posterior_predictive.

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs), or az.InferenceData if return_idata=True.
        """
        self._check_fitted()
        assert self.model is not None  # guaranteed by _check_fitted
        X = self._check_n_features(X)

        n_obs = X.shape[0]
        X_aug = self._augment_X(X)
        dummy_t = np.ones(n_obs, dtype=float)
        upper = np.full(n_obs, np.inf, dtype=float)

        with self.model:
            pm.set_data({"X_aug": X_aug, "t_obs": dummy_t, "upper": upper})
            idata_ppc = pm.sample_posterior_predictive(
                self.idata,
                var_names=["obs"],
                extend_inferencedata=False,
                random_seed=random_seed,
                **sample_kwargs,
            )

        if return_idata:
            return idata_ppc

        return self._flatten_ppc(idata_ppc)  # (n_samples, n_obs)

    def _check_fitted(self) -> None:
        if self.idata is None or self.model is None:
            raise RuntimeError("Model has not been fitted yet — call fit() first.")
