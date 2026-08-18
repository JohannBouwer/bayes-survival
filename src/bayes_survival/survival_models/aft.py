from __future__ import annotations
from typing import ClassVar

import arviz as az
import numpy as np
import pymc as pm
from scipy.special import expit, ndtr

from .base import BaseSurvivalModel, PriorSpec


class WeibullAFTModel(BaseSurvivalModel):
    """Weibull Accelerated Failure Time model.

    Survival function:
        S(t | x) = exp( -(t / exp(Xβ))^α )

    An intercept is added automatically; users pass raw covariates only.
    Positive β_j corresponds to stochastically longer survival times.
    """

    default_priors: ClassVar[PriorSpec] = {
        "alpha": (pm.Gamma, {"alpha": 5, "beta": 2}),
        "beta": (pm.Normal, {"mu": 0, "sigma": 5}),
    }

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        self._n_features = X.shape[1]
        X_aug = self._augment_X(X)  # (n_obs, n_features + 1)
        upper = np.where(event == 1, np.inf, t).astype(float)

        with pm.Model() as model:
            X_data = pm.Data("X_aug", X_aug)
            t_data = pm.Data("t_obs", t.astype(float))
            up_data = pm.Data("upper", upper)

            alpha = self._prior("alpha")
            beta = self._prior("beta", shape=X_aug.shape[1])

            lam = pm.math.exp(pm.math.dot(X_data, beta))  # scale per observation

            pm.Censored(
                "obs",
                pm.Weibull.dist(alpha=alpha, beta=lam),
                lower=None,
                upper=up_data,
                observed=t_data,
            )

        return model

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = self._check_n_features(X)

        X_aug = self._augment_X(X)  # (n_obs, n_features + 1)

        assert self.idata is not None
        posterior = self.idata.posterior
        beta_samples = posterior["beta"].values  # (chains, draws, n_coef)
        alpha_samples = posterior["alpha"].values  # (chains, draws)

        n_chains, n_draws, n_coef = beta_samples.shape
        n_samples = n_chains * n_draws

        beta_flat = beta_samples.reshape(n_samples, n_coef)  # (n_samples, n_coef)
        alpha_flat = alpha_samples.reshape(n_samples)  # (n_samples,)

        mu = beta_flat @ X_aug.T  # (n_samples, n_obs)
        lam = np.exp(mu)  # (n_samples, n_obs)

        t_exp = times[np.newaxis, np.newaxis, :]  # (1, 1, n_times)
        lam_exp = lam[:, :, np.newaxis]  # (n_samples, n_obs, 1)
        alpha_exp = alpha_flat[:, np.newaxis, np.newaxis]  # (n_samples, 1, 1)

        return np.exp(-((t_exp / lam_exp) ** alpha_exp))  # (n_samples, n_obs, n_times)


class LogNormalAFTModel(BaseSurvivalModel):
    """Log-Normal Accelerated Failure Time model.

    Survival function:
        S(t | x) = 1 - Φ((log(t) - Xβ) / σ)  =  Φ((Xβ - log(t)) / σ)

    An intercept is added automatically; users pass raw covariates only.
    Positive β_j corresponds to stochastically longer survival times.
    """

    default_priors: ClassVar[PriorSpec] = {
        "sigma": (pm.Gamma, {"alpha": 5, "beta": 2}),
        "beta": (pm.Normal, {"mu": 0, "sigma": 5}),
    }

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        self._n_features = X.shape[1]
        X_aug = self._augment_X(X)  # (n_obs, n_features + 1)
        upper = np.where(event == 1, np.inf, t).astype(float)

        with pm.Model() as model:
            X_data = pm.Data("X_aug", X_aug)
            t_data = pm.Data("t_obs", t.astype(float))
            up_data = pm.Data("upper", upper)

            sigma = self._prior("sigma")
            beta = self._prior("beta", shape=X_aug.shape[1])

            mu_pred = pm.math.dot(X_data, beta)  # log-scale mean per observation

            pm.Censored(
                "obs",
                pm.LogNormal.dist(mu=mu_pred, sigma=sigma),
                lower=None,
                upper=up_data,
                observed=t_data,
            )

        return model

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = self._check_n_features(X)

        X_aug = self._augment_X(X)  # (n_obs, n_features + 1)

        assert self.idata is not None
        posterior = self.idata.posterior
        beta_samples = posterior["beta"].values  # (chains, draws, n_coef)
        sigma_samples = posterior["sigma"].values  # (chains, draws)

        n_chains, n_draws, n_coef = beta_samples.shape
        n_samples = n_chains * n_draws

        beta_flat = beta_samples.reshape(n_samples, n_coef)  # (n_samples, n_coef)
        sigma_flat = sigma_samples.reshape(n_samples)  # (n_samples,)

        mu = beta_flat @ X_aug.T  # (n_samples, n_obs)
        log_t = np.log(times)  # (n_times,)

        # z shape: (n_samples, n_obs, n_times)
        z = (log_t[np.newaxis, np.newaxis, :] - mu[:, :, np.newaxis]) / sigma_flat[
            :, np.newaxis, np.newaxis
        ]

        return ndtr(
            -z
        )  # S(t|x) = 1 - Φ(z) = Φ(-z),  shape: (n_samples, n_obs, n_times)


class LogLogisticAFTModel(BaseSurvivalModel):
    """Log-Logistic Accelerated Failure Time model.

    Survival function:
        S(t | x) = 1 / (1 + (t / exp(Xβ))^α)

    An intercept is added automatically; users pass raw covariates only.
    Positive β_j corresponds to stochastically longer survival times.
    The hazard is non-monotonic (rises then falls), making this model
    suitable when event rates peak at some intermediate time.
    """

    default_priors: ClassVar[PriorSpec] = {
        "alpha": (pm.Gamma, {"alpha": 5, "beta": 2}),
        "beta": (pm.Normal, {"mu": 0, "sigma": 5}),
    }

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        """Build the log-logistic model via the log-time re-parametrisation.

        log(T) | x ~ Logistic(mu=Xβ, s=1/α), so T is log-logistic.
        We model log(T) directly with pm.Logistic, censoring on the log scale.
        The Jacobian constant (−Σ log t for uncensored rows) is omitted here: it
        does not depend on any parameters, so it has no effect on the posterior
        shape or MCMC samples.

        It *is* however applied in ``_attach_log_likelihood``, because the
        reported log-likelihood has to be on the time scale to be comparable
        with the other models under ``az.compare``.
        """
        self._n_features = X.shape[1]
        X_aug = self._augment_X(X)  # (n_obs, n_features + 1)

        log_t = np.log(t).astype(float)
        # Right-censored rows: upper bound on the log scale; uncensored: inf
        log_upper = np.where(event == 1, np.inf, log_t)
        self._store_jacobian(log_t, event)

        with pm.Model() as model:
            X_data = pm.Data("X_aug", X_aug)
            logt_data = pm.Data("t_obs", log_t)
            up_data = pm.Data("upper", log_upper)

            alpha = self._prior("alpha")  # shape parameter (α > 0)
            beta = self._prior("beta", shape=X_aug.shape[1])

            mu_pred = pm.math.dot(X_data, beta)  # log-scale location per obs
            s = 1.0 / alpha  # logistic scale on the log-time axis

            # Model log(T) ~ Logistic(mu, s), censored on the log scale
            pm.Censored(
                "obs",
                pm.Logistic.dist(mu=mu_pred, s=s),
                lower=None,
                upper=up_data,
                observed=logt_data,
            )

        return model

    def _store_jacobian(self, log_t: np.ndarray, event: np.ndarray) -> None:
        """Record the change-of-variables term needed to score on the time scale."""
        self._logt_jacobian = log_t * (np.asarray(event) == 1)

    def _attach_log_likelihood(self) -> None:
        """Attach a log_likelihood measured on the time scale, not the log-time scale.

        ``build_model`` fits ``log(T)``, so ``pm.compute_log_likelihood`` returns
        ``log f_{log T}``. The change of variables ``T = exp(Y)`` gives

            log f_T(t) = log f_{log T}(log t) - log t

        with no correction for censored rows, since survival probabilities are
        invariant under a monotone transform.

        Omitting this term is harmless for sampling — it is constant in the
        parameters, which is why ``build_model`` drops it — but it would offset
        this model's elpd by ``+sum(log t)`` over the uncensored rows relative to
        every other model, making ``az.compare`` against them silently wrong.
        """
        super()._attach_log_likelihood()

        assert self.idata is not None
        ll = self.idata.log_likelihood["obs"]
        self.idata.log_likelihood["obs"] = (ll.dims, ll.values - self._logt_jacobian)

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = self._check_n_features(X)

        X_aug = self._augment_X(X)  # (n_obs, n_features + 1)

        assert self.idata is not None
        posterior = self.idata.posterior
        beta_samples = posterior["beta"].values  # (chains, draws, n_coef)
        alpha_samples = posterior["alpha"].values  # (chains, draws)

        n_chains, n_draws, n_coef = beta_samples.shape
        n_samples = n_chains * n_draws

        beta_flat = beta_samples.reshape(n_samples, n_coef)  # (n_samples, n_coef)
        alpha_flat = alpha_samples.reshape(n_samples)  # (n_samples,)

        mu = beta_flat @ X_aug.T  # (n_samples, n_obs)
        lam = np.exp(mu)  # (n_samples, n_obs)

        t_exp = times[np.newaxis, np.newaxis, :]  # (1, 1, n_times)
        lam_exp = lam[:, :, np.newaxis]  # (n_samples, n_obs, 1)
        alpha_exp = alpha_flat[:, np.newaxis, np.newaxis]  # (n_samples, 1, 1)

        # S(t|x) = expit(α * (log λ - log t))
        return expit(
            alpha_exp * (np.log(lam_exp) - np.log(t_exp))
        )  # (n_samples, n_obs, n_times)

    def sample_predicted_event_times(
        self,
        X: np.ndarray,
        return_idata: bool = False,
        random_seed: int | None = None,
        **sample_kwargs,
    ) -> np.ndarray | az.InferenceData:
        """Draw posterior-predictive event times on the original time scale.

        ``build_model`` models ``log(T)`` rather than ``T``, so the "obs" variable
        the base implementation samples lives on the log-time axis. Without this
        override the caller would receive log-times — including negative values,
        which are impossible for a survival time. Exponentiating recovers ``T``.

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        return_idata : if True, return the raw az.InferenceData object (also
            converted to the original time scale).
        random_seed : seed for the predictive draws; pass an int for reproducibility.
        **sample_kwargs : forwarded to pm.sample_posterior_predictive.

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs), or az.InferenceData if return_idata=True.
        """
        idata_ppc = super().sample_predicted_event_times(
            X,
            return_idata=True,
            random_seed=random_seed,
            **sample_kwargs,
        )
        # Convert log(T) draws back to T, in place, so both return paths agree.
        idata_ppc.posterior_predictive["obs"] = np.exp(
            idata_ppc.posterior_predictive["obs"]
        )

        if return_idata:
            return idata_ppc

        return self._flatten_ppc(idata_ppc)  # (n_samples, n_obs)
