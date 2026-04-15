from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .aft import LogLogisticAFTModel, LogNormalAFTModel, WeibullAFTModel
from .base import HierarchySpec, PriorSpec


def _build_hierarchical_beta(
    hierarchies: list[HierarchySpec],
    feature_names: list[str],
    n_coef: int,
    flat_prior: tuple[type, dict],
) -> pt.TensorVariable:
    """Assemble the full coefficient vector inside an active ``pm.Model`` context.

    Assign hierarchical priors and flat priors to specified variables.

    The returned Deterministic tensor ``"beta"`` has shape ``(n_coef,)`` and
    mirrors the layout of the augmented design matrix ``X_aug``:

    * index 0            → intercept  (flat prior)
    * index ``j + 1``    → feature ``j`` — either a flat prior or drawn from a
                           group-specific Normal(mu_group, sigma_group).

    Parameters
    ----------
    hierarchies:
        List of :class:`~bayes_survival.survival_models.base.HierarchySpec`
        objects, each describing one group of covariates and its hyper-priors.
    feature_names:
        Ordered list of column names corresponding to the raw (un-augmented)
        feature matrix ``X``.  Set from ``df.columns`` before calling
        ``build_model``.
    n_coef:
        Total number of coefficients, i.e. ``len(feature_names) + 1``.
    flat_prior:
        ``(DistributionClass, kwargs)`` tuple taken from ``self.priors["beta"]``.
        Used for the intercept and all ungrouped covariates.

    Returns
    -------
    pt.TensorVariable
        A ``pm.Deterministic`` named ``"beta"`` of shape ``(n_coef,)``.
    """
    n_features = len(feature_names)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    # validate hierarchies
    seen_names: set[str] = set()
    seen_group_names: set[str] = set()
    for grp in hierarchies:
        if grp.name in seen_group_names:
            raise ValueError(f"Duplicate HierarchySpec name: '{grp.name}'")
        seen_group_names.add(grp.name)
        for cname in grp.covariate_names:
            if cname not in name_to_idx:
                raise ValueError(
                    f"HierarchySpec '{grp.name}': covariate '{cname}' not found "
                    f"in feature_names {feature_names}"
                )
            if cname in seen_names:
                raise ValueError(
                    f"Covariate '{cname}' appears in more than one HierarchySpec"
                )
            seen_names.add(cname)

    # determine which raw-feature indices are grouped
    grouped_feature_indices: set[int] = {
        name_to_idx[cname] for grp in hierarchies for cname in grp.covariate_names
    }
    ungrouped_feature_indices: list[int] = sorted(
        set(range(n_features)) - grouped_feature_indices
    )

    # build per-column tensors
    # parts[k] will hold a rank-1 tensor of length 1 for X_aug column k
    parts: list[pt.TensorVariable] = [None] * n_coef  # type: ignore[list-item]

    flat_dist_cls, flat_kwargs = flat_prior

    # intercept (X_aug col 0) — always flat
    beta_intercept = flat_dist_cls("beta_intercept", **flat_kwargs)
    parts[0] = beta_intercept[None]  # (1,)

    # ungrouped covariates — one shared flat variable, sliced
    if ungrouped_feature_indices:
        n_ung = len(ungrouped_feature_indices)
        beta_ung = flat_dist_cls("beta_ungrouped", **{**flat_kwargs, "shape": n_ung})
        for local_i, feat_i in enumerate(ungrouped_feature_indices):
            parts[feat_i + 1] = beta_ung[local_i : local_i + 1]

    # hierarchical groups
    for grp in hierarchies:
        mu_dist_cls, mu_kwargs = grp.mu_prior
        sigma_dist_cls, sigma_kwargs = grp.sigma_prior
        n_g = len(grp.covariate_names)

        mu_h = mu_dist_cls(f"mu_{grp.name}", **mu_kwargs)
        sigma_h = sigma_dist_cls(f"sigma_{grp.name}", **sigma_kwargs)

        if grp.centered:
            beta_h = pm.Normal(f"beta_{grp.name}", mu=mu_h, sigma=sigma_h, shape=n_g)
        else:
            beta_raw = pm.Normal(f"beta_{grp.name}_raw", mu=0, sigma=1, shape=n_g)
            beta_h = pm.Deterministic(f"beta_{grp.name}", mu_h + sigma_h * beta_raw)

        for local_i, cname in enumerate(grp.covariate_names):
            feat_i = name_to_idx[cname]
            parts[feat_i + 1] = beta_h[local_i : local_i + 1]

    assert all(p is not None for p in parts), "Some beta positions were not assigned."

    return pm.Deterministic("beta", pt.concatenate(parts))


def _extract_array(
    X: pd.DataFrame | np.ndarray, feature_names: list[str]
) -> np.ndarray:
    """Convert a DataFrame (or validate an ndarray) to a float ndarray.

    Other models take numpy array, the hierarchical model takes a dataframe with columns for groupings.

    If *X* is a DataFrame its column order is reindexed to match *feature_names*
    so that callers can pass DataFrames with different column orderings during
    prediction.
    """
    if isinstance(X, pd.DataFrame):
        missing = set(feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")
        return X[feature_names].values.astype(float)
    return np.asarray(X, dtype=float)


class HierarchicalWeibullAFTModel(WeibullAFTModel):
    """Weibull AFT model with user-defined hierarchical (partial-pooling) priors.

    Covariates belonging to a :class:`~bayes_survival.survival_models.base.HierarchySpec`
    group are drawn from ``Normal(mu_group, sigma_group)``; the hyper-parameters
    ``mu_group`` and ``sigma_group`` are given their own priors, enabling
    information sharing within the group.  All other covariates (and the
    intercept) use the flat ``"beta"`` prior inherited from
    :class:`WeibullAFTModel`.

    Parameters
    ----------
    hierarchies:
        One or more :class:`HierarchySpec` objects defining the groups.
    priors:
        Optional overrides for model-level priors (``"alpha"``, ``"beta"``).
        See :class:`WeibullAFTModel` for details.

    Notes
    -----
    ``fit()`` requires a ``pd.DataFrame`` for *X* so that covariate names can
    be resolved.  All prediction methods accept either a DataFrame or a plain
    NumPy array (with columns in the same order as the training DataFrame).
    """

    def __init__(
        self,
        hierarchies: list[HierarchySpec],
        priors: PriorSpec | None = None,
    ) -> None:
        super().__init__(priors=priors)
        self.hierarchies = hierarchies
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        t: np.ndarray,
        event: np.ndarray,
        draws: int = 1000,
        tune: int = 1000,
        nuts_sampler: str = "nutpie",
        **sample_kwargs,
    ) -> HierarchicalWeibullAFTModel:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "HierarchicalWeibullAFTModel.fit() requires a pd.DataFrame for X "
                "so that covariate names can be resolved for the hierarchy groups."
            )
        self.feature_names_ = list(X.columns)
        return super().fit(
            X.values.astype(float),
            t,
            event,
            draws=draws,
            tune=tune,
            nuts_sampler=nuts_sampler,
            **sample_kwargs,
        )

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        self._n_features = X.shape[1]
        X_aug = self._augment_X(X)
        upper = np.where(event == 1, np.inf, t).astype(float)

        with pm.Model() as model:
            X_data = pm.Data("X_aug", X_aug)
            t_data = pm.Data("t_obs", t.astype(float))
            up_data = pm.Data("upper", upper)

            alpha = self._prior("alpha")
            beta = _build_hierarchical_beta(
                self.hierarchies,
                self.feature_names_,  # type: ignore[arg-type]
                X_aug.shape[1],
                self.priors["beta"],
            )

            lam = pm.math.exp(pm.math.dot(X_data, beta))
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
        X: pd.DataFrame | np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        arr = _extract_array(X, self.feature_names_ or [])
        return super()._predict_survival_samples(arr, times)

    def sample_predicted_event_times(
        self,
        X: pd.DataFrame | np.ndarray,
        return_idata: bool = False,
        **sample_kwargs,
    ) -> np.ndarray:
        arr = _extract_array(X, self.feature_names_ or [])
        return super().sample_predicted_event_times(
            arr, return_idata=return_idata, **sample_kwargs
        )


class HierarchicalLogNormalAFTModel(LogNormalAFTModel):
    """Log-Normal AFT model with user-defined hierarchical (partial-pooling) priors.

    See :class:`HierarchicalWeibullAFTModel` for full documentation; the only
    difference is that the event-time distribution is Log-Normal.
    """

    def __init__(
        self,
        hierarchies: list[HierarchySpec],
        priors: PriorSpec | None = None,
    ) -> None:
        super().__init__(priors=priors)
        self.hierarchies = hierarchies
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        t: np.ndarray,
        event: np.ndarray,
        draws: int = 1000,
        tune: int = 1000,
        nuts_sampler: str = "nutpie",
        **sample_kwargs,
    ) -> HierarchicalLogNormalAFTModel:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "HierarchicalLogNormalAFTModel.fit() requires a pd.DataFrame for X "
                "so that covariate names can be resolved for the hierarchy groups."
            )
        self.feature_names_ = list(X.columns)
        return super().fit(
            X.values.astype(float),
            t,
            event,
            draws=draws,
            tune=tune,
            nuts_sampler=nuts_sampler,
            **sample_kwargs,
        )

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        self._n_features = X.shape[1]
        X_aug = self._augment_X(X)
        upper = np.where(event == 1, np.inf, t).astype(float)

        with pm.Model() as model:
            X_data = pm.Data("X_aug", X_aug)
            t_data = pm.Data("t_obs", t.astype(float))
            up_data = pm.Data("upper", upper)

            sigma = self._prior("sigma")
            beta = _build_hierarchical_beta(
                self.hierarchies,
                self.feature_names_,  # type: ignore[arg-type]
                X_aug.shape[1],
                self.priors["beta"],
            )

            mu_pred = pm.math.dot(X_data, beta)
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
        X: pd.DataFrame | np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        arr = _extract_array(X, self.feature_names_ or [])
        return super()._predict_survival_samples(arr, times)

    def sample_predicted_event_times(
        self,
        X: pd.DataFrame | np.ndarray,
        return_idata: bool = False,
        **sample_kwargs,
    ) -> np.ndarray:
        arr = _extract_array(X, self.feature_names_ or [])
        return super().sample_predicted_event_times(
            arr, return_idata=return_idata, **sample_kwargs
        )


class HierarchicalLogLogisticAFTModel(LogLogisticAFTModel):
    """Log-Logistic AFT model with user-defined hierarchical (partial-pooling) priors.

    See :class:`HierarchicalWeibullAFTModel` for full documentation; the only
    difference is that the event-time distribution is Log-Logistic.
    """

    def __init__(
        self,
        hierarchies: list[HierarchySpec],
        priors: PriorSpec | None = None,
    ) -> None:
        super().__init__(priors=priors)
        self.hierarchies = hierarchies
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        t: np.ndarray,
        event: np.ndarray,
        draws: int = 1000,
        tune: int = 1000,
        nuts_sampler: str = "nutpie",
        **sample_kwargs,
    ) -> HierarchicalLogLogisticAFTModel:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "HierarchicalLogLogisticAFTModel.fit() requires a pd.DataFrame for X "
                "so that covariate names can be resolved for the hierarchy groups."
            )
        self.feature_names_ = list(X.columns)
        return super().fit(
            X.values.astype(float),
            t,
            event,
            draws=draws,
            tune=tune,
            nuts_sampler=nuts_sampler,
            **sample_kwargs,
        )

    def build_model(
        self,
        X: np.ndarray,
        t: np.ndarray,
        event: np.ndarray,
    ) -> pm.Model:
        self._n_features = X.shape[1]
        X_aug = self._augment_X(X)
        log_t = np.log(t).astype(float)
        log_upper = np.where(event == 1, np.inf, log_t)

        with pm.Model() as model:
            X_data = pm.Data("X_aug", X_aug)
            logt_data = pm.Data("t_obs", log_t)
            up_data = pm.Data("upper", log_upper)

            alpha = self._prior("alpha")
            beta = _build_hierarchical_beta(
                self.hierarchies,
                self.feature_names_,  # type: ignore[arg-type]
                X_aug.shape[1],
                self.priors["beta"],
            )

            mu_pred = pm.math.dot(X_data, beta)
            s = 1.0 / alpha
            pm.Censored(
                "obs",
                pm.Logistic.dist(mu=mu_pred, s=s),
                lower=None,
                upper=up_data,
                observed=logt_data,
            )

        return model

    def _predict_survival_samples(
        self,
        X: pd.DataFrame | np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        arr = _extract_array(X, self.feature_names_ or [])
        return super()._predict_survival_samples(arr, times)

    def sample_predicted_event_times(
        self,
        X: pd.DataFrame | np.ndarray,
        return_idata: bool = False,
        **sample_kwargs,
    ) -> np.ndarray:
        arr = _extract_array(X, self.feature_names_ or [])
        return super().sample_predicted_event_times(
            arr, return_idata=return_idata, **sample_kwargs
        )
