"""Shared input validation for survival models.

Every model entry point funnels its raw ``(X, t, event)`` through
:func:`validate_survival_inputs` so that malformed data fails immediately with a
readable message, rather than surfacing later as a ``-inf`` log-likelihood, a NaN
gradient, or a silently dropped observation.
"""

from __future__ import annotations

import numpy as np


def _prefix(model_name: str) -> str:
    return f"{model_name}: " if model_name else ""


def _as_float_array(value, name: str, prefix: str) -> np.ndarray:
    """Coerce to a float64 ndarray, re-raising conversion errors readably."""
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{prefix}`{name}` must be numeric and convertible to a float array; "
            f"got {type(value).__name__}."
        ) from exc


def validate_survival_inputs(
    t,
    event,
    X=None,
    *,
    require_positive_t: bool = True,
    model_name: str = "",
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Validate and normalise survival inputs.

    Parameters
    ----------
    t : array-like of shape (n_obs,)
        Observed times — the event time, or the last follow-up time for censored
        observations.
    event : array-like of shape (n_obs,)
        Event indicator: ``1`` = event observed, ``0`` = right-censored.
    X : array-like of shape (n_obs, n_features), optional
        Covariate matrix without an intercept column. Pass ``None`` for models
        that take no covariates (the nonparametric estimators).
    require_positive_t : bool
        If ``True`` (the default) require ``t > 0``. Models that work on the
        log-time scale, and the piecewise Cox model — where a subject with
        ``t == 0`` contributes no at-risk interval at all — need this. The
        conjugate nonparametric estimators are well defined at ``t == 0`` and
        pass ``False``.
    model_name : str
        Prepended to error messages so the caller knows which model rejected the
        input.

    Returns
    -------
    X : np.ndarray of shape (n_obs, n_features), or None
    t : np.ndarray of shape (n_obs,), float64
    event : np.ndarray of shape (n_obs,), float64

    Raises
    ------
    TypeError
        If an argument cannot be interpreted as a numeric array.
    ValueError
        If shapes disagree, values are non-finite, ``event`` is not binary, or
        ``t`` violates the positivity requirement.
    """
    prefix = _prefix(model_name)

    t = _as_float_array(t, "t", prefix)
    event = _as_float_array(event, "event", prefix)

    if t.ndim != 1:
        raise ValueError(f"{prefix}`t` must be 1-D, got shape {t.shape}.")
    if event.ndim != 1:
        raise ValueError(f"{prefix}`event` must be 1-D, got shape {event.shape}.")
    if t.size == 0:
        raise ValueError(f"{prefix}`t` is empty; there is nothing to fit.")
    if t.shape[0] != event.shape[0]:
        raise ValueError(
            f"{prefix}`t` and `event` must have the same length, "
            f"got {t.shape[0]} and {event.shape[0]}."
        )

    if not np.all(np.isfinite(t)):
        n_bad = int((~np.isfinite(t)).sum())
        raise ValueError(
            f"{prefix}`t` contains {n_bad} non-finite value(s) (NaN or inf)."
        )

    if require_positive_t:
        bad = t <= 0
        if bad.any():
            raise ValueError(
                f"{prefix}`t` must be strictly positive — found {int(bad.sum())} "
                f"value(s) <= 0 (minimum {t.min()}). This model evaluates log(t) or "
                "t**alpha, so a non-positive time yields an undefined log-likelihood."
            )
    else:
        bad = t < 0
        if bad.any():
            raise ValueError(
                f"{prefix}`t` must be non-negative — found {int(bad.sum())} "
                f"negative value(s) (minimum {t.min()})."
            )

    invalid = ~np.isin(event, (0.0, 1.0))
    if invalid.any():
        observed = np.unique(event[invalid]).tolist()
        raise ValueError(
            f"{prefix}`event` must contain only 0 (censored) and 1 (event); "
            f"found {observed}."
        )

    if X is None:
        return None, t, event

    X = _as_float_array(X, "X", prefix)

    if X.ndim == 1:
        raise ValueError(
            f"{prefix}`X` must be 2-D (n_obs, n_features), got 1-D shape {X.shape}. "
            "For a single covariate, pass X.reshape(-1, 1)."
        )
    if X.ndim != 2:
        raise ValueError(
            f"{prefix}`X` must be 2-D (n_obs, n_features), got shape {X.shape}."
        )
    if X.shape[0] != t.shape[0]:
        raise ValueError(
            f"{prefix}`X` has {X.shape[0]} row(s) but `t` has {t.shape[0]} entry/entries."
        )
    if X.shape[1] == 0:
        raise ValueError(f"{prefix}`X` must have at least one column.")
    if not np.all(np.isfinite(X)):
        n_bad = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"{prefix}`X` contains {n_bad} non-finite value(s) (NaN or inf)."
        )

    return X, t, event
