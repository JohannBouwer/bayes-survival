from __future__ import annotations
from typing import ClassVar

import numpy as np
import pymc as pm

from .base import BaseSurvivalModel, PriorSpec


class PiecewiseCoxPHModel(BaseSurvivalModel):
    """Piecewise Constant Bayesian Cox Proportional Hazards Model.

    The hazard function is:
        h(t | x) = h_0(t) * exp(X @ beta)

    where h_0(t) is piecewise constant over user-defined (or data-driven)
    intervals and log(h_0) follows a GaussianRandomWalk across intervals.

    Fitting uses the Poisson likelihood equivalence: data are expanded into
    long format (one row per observation-interval pair while at risk) and the
    event count in each cell is modelled as Poisson with rate h_k * exp(X @ beta)
    and exposure offset equal to time at risk in that interval.

    No intercept is included in beta; the baseline hazard absorbs it.

    Parameters
    ----------
    n_intervals : int, optional
        Number of intervals. Interior cut points are placed at evenly-spaced
        quantiles of the *event* times (censored observations excluded).
        Exactly one of n_intervals or cuts must be provided.
    cuts : list of float, optional
        Explicit interior cut points (positive, strictly increasing).
        Exactly one of n_intervals or cuts must be provided.
    priors : PriorSpec, optional
        Override default priors. Valid keys: "grw_sigma", "beta".
    """

    default_priors: ClassVar[PriorSpec] = {
        "grw_sigma": (pm.HalfNormal, {"sigma": 1.0}),
        "beta": (pm.Normal, {"mu": 0, "sigma": 1}),
    }

    def __init__(
        self,
        n_intervals: int | None = None,
        cuts: list[float] | None = None,
        priors: PriorSpec | None = None,
    ) -> None:
        if (n_intervals is None) == (cuts is None):
            raise ValueError("Provide exactly one of n_intervals or cuts.")
        self.n_intervals = n_intervals
        self.cuts = cuts
        self._cuts: np.ndarray | None = (
            None  # computed at fit time, used for prediction
        )
        super().__init__(priors=priors)

    @staticmethod
    def _compute_cuts(
        t: np.ndarray,
        event: np.ndarray,
        n_intervals: int,
    ) -> np.ndarray:
        """Compute interior cut points from quantiles of event times."""
        event_times = t[event == 1]
        if len(event_times) == 0:
            raise ValueError("No observed events found; cannot compute cut points.")
        q = np.linspace(0, 1, n_intervals + 1)[1:-1]
        cuts = np.unique(np.quantile(event_times, q))
        return cuts.astype(float)

    @staticmethod
    def _expand_data(
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
        cuts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Expand (X, t, event) to long format for the Poisson likelihood.

        For each observation i and each interval k in which i is at risk,
        one row is added with:
          - exposure : time spent in interval k
          - event_ik : 1 iff observation i had an event in interval k, else 0
          - interval_idx : k (for indexing into log_baseline)

        Parameters
        ----------
        X : (n_obs, n_features)
        t : (n_obs,)
        event : (n_obs,) — 1 = event, 0 = censored
        cuts : interior cut points (sorted, positive)

        Returns
        -------
        X_long       : (N_long, n_features)
        interval_idx : (N_long,) int
        exposure     : (N_long,) float — time at risk in interval
        events_long  : (N_long,) int   — 0/1 event indicator per row
        """
        boundaries = np.concatenate([[0.0], cuts, [np.inf]])
        n_intervals = len(boundaries) - 1

        X_rows: list[np.ndarray] = []
        idx_rows: list[int] = []
        exp_rows: list[float] = []
        ev_rows: list[int] = []

        for i in range(len(t)):
            for k in range(n_intervals):
                left = boundaries[k]
                right = boundaries[k + 1]
                if t[i] <= left:
                    break  # obs i not at risk in this or any later interval
                exposure = float(min(t[i], right) - left)
                ev = int(event[i] == 1 and t[i] <= right)
                X_rows.append(X[i])
                idx_rows.append(k)
                exp_rows.append(exposure)
                ev_rows.append(ev)

        return (
            np.array(X_rows, dtype=float),
            np.array(idx_rows, dtype=int),
            np.array(exp_rows, dtype=float),
            np.array(ev_rows, dtype=int),
        )

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        self._n_features = X.shape[1]

        # Determine cut points
        if self.n_intervals is not None:
            cuts = self._compute_cuts(t, event, self.n_intervals)
        else:
            cuts = np.asarray(self.cuts, dtype=float)
            if not np.all(cuts > 0) or not np.all(np.diff(cuts) > 0):
                raise ValueError("cuts must be strictly increasing positive values.")
        self._cuts = cuts
        n_intervals = len(cuts) + 1

        # Expand to long format
        X_long, interval_idx, exposure, events_long = self._expand_data(
            X, t, event, cuts
        )
        log_exposure = np.log(exposure)

        with pm.Model() as model:
            grw_sigma = self._prior("grw_sigma")
            log_baseline = pm.GaussianRandomWalk(
                "log_baseline", sigma=grw_sigma, shape=n_intervals
            )
            beta = self._prior("beta", shape=X.shape[1])

            # Poisson rate: h_k * exp(X @ beta) * exposure
            log_mu = (
                log_baseline[interval_idx] + pm.math.dot(X_long, beta) + log_exposure
            )
            pm.Poisson("obs", mu=pm.math.exp(log_mu), observed=events_long)

        return model

    def _predict_survival_samples(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Return survival samples of shape (n_samples, n_obs, n_times)."""
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")
        if self._cuts is None:
            raise RuntimeError("Model has not been fitted yet — call fit() first.")

        assert self.idata is not None
        posterior = self.idata.posterior
        lb_samples = posterior["log_baseline"].values  # (chains, draws, n_intervals)
        beta_samples = posterior["beta"].values  # (chains, draws, n_features)

        n_chains, n_draws, n_intervals = lb_samples.shape
        n_samples = n_chains * n_draws

        h0_flat = np.exp(lb_samples.reshape(n_samples, n_intervals))  # (S, K)
        beta_flat = beta_samples.reshape(n_samples, -1)  # (S, p)

        # Interval boundaries
        boundaries = np.concatenate([[0.0], self._cuts, [np.inf]])
        lefts = boundaries[:-1]  # (K,)
        rights = boundaries[1:]  # (K,) — last entry is inf

        # Exposure per (time, interval): max(0, min(t, right) - left)
        # Shape: (n_times, K)
        t_col = times[:, np.newaxis]  # (n_times, 1)
        exposures = np.maximum(0.0, np.minimum(t_col, rights) - lefts)  # (n_times, K)

        # Cumulative baseline hazard per sample and time: (S, n_times)
        cum_H0 = h0_flat @ exposures.T  # (S, K) @ (K, n_times) = (S, n_times)

        # Hazard ratio per obs: (S, n_obs)
        hr = np.exp(beta_flat @ X.T)

        # H(t|x) = H0(t) * hr(x),  shape: (S, n_obs, n_times)
        H = cum_H0[:, np.newaxis, :] * hr[:, :, np.newaxis]

        return np.exp(-H)  # (S, n_obs, n_times)

    def sample_predicted_event_times(
        self,
        X: np.ndarray,
        return_idata: bool = False,
        **sample_kwargs,
    ) -> np.ndarray:
        """Draw posterior-predictive event times via piecewise-exponential inverse CDF.

        Parameters
        ----------
        X : (n_obs, n_features) — raw covariates, no intercept column.
        return_idata : ignored (kept for interface compatibility).

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs) — sampled event times.
        """
        self._check_fitted()
        if self._cuts is None:
            raise RuntimeError("Model has not been fitted yet — call fit() first.")
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")

        assert self.idata is not None
        posterior = self.idata.posterior
        lb_samples = posterior["log_baseline"].values  # (chains, draws, n_intervals)
        beta_samples = posterior["beta"].values  # (chains, draws, n_features)

        n_chains, n_draws, n_intervals = lb_samples.shape
        n_samples = n_chains * n_draws

        h0_flat = np.exp(lb_samples.reshape(n_samples, n_intervals))  # (S, K)
        beta_flat = beta_samples.reshape(n_samples, -1)  # (S, p)

        # Hazard rate per sample, obs, interval: (S, n_obs, K)
        hr = np.exp(beta_flat @ X.T)  # (S, n_obs)
        rates = h0_flat[:, np.newaxis, :] * hr[:, :, np.newaxis]  # (S, n_obs, K)

        boundaries = np.concatenate([[0.0], self._cuts, [np.inf]])
        widths = np.diff(boundaries)  # (K,) — last entry is inf

        # Target cumulative hazard drawn from Exp(1) ≡ -log(Uniform(0,1))
        rng = np.random.default_rng()
        u = rng.uniform(0.0, 1.0, (n_samples, X.shape[0]))
        target = -np.log(u)  # (S, n_obs)

        event_times = np.full((n_samples, X.shape[0]), np.inf)
        cum_hazard = np.zeros((n_samples, X.shape[0]))

        for k in range(n_intervals):
            left = boundaries[k]
            width = widths[k]  # inf for last interval
            rate_k = rates[:, :, k]  # (S, n_obs)

            interval_contribution = rate_k * width  # inf when k == last
            cum_hazard_end = cum_hazard + interval_contribution

            # Observations whose target falls in this interval
            in_interval = (
                (target > cum_hazard)
                & (target <= cum_hazard_end)
                & np.isinf(event_times)
            )

            # Solve cum_hazard + rate_k * (t - left) = target  =>  t = left + residual/rate
            safe_rate = np.where(rate_k > 0, rate_k, 1e-300)
            t_in = left + (target - cum_hazard) / safe_rate
            event_times = np.where(in_interval, t_in, event_times)

            cum_hazard = cum_hazard_end

            if not np.any(np.isinf(event_times)):
                break

        return event_times  # (n_samples, n_obs)
