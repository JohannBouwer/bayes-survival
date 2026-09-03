"""Held-out evaluation metrics for right-censored survival predictions.

LOO (``az.compare``) answers "which model fits the data I have?". These answer
"does it work on subjects it never saw?" — a different question, and the two can
disagree.

Every function here takes the same pair: a predicted survival matrix ``surv`` of
shape ``(n_obs, n_times)`` together with the ``time_grid`` it was evaluated on.
One prediction call therefore feeds every metric, and horizons are resolved from
the grid internally. ``numpy.interp`` is exact at knots, so putting the horizons
into the grid (``np.union1d(grid, horizons)``) makes the Brier scores exact
rather than interpolated.

Harrell's concordance is re-exported from lifelines rather than reimplemented —
theirs is the canonical implementation. The Kaplan-Meier estimator used for the
censoring distribution is lifelines' as well: this package's own
``KaplanMeierModel`` is *Bayesian* (Beta posterior), not the product-limit
estimator, so using it here would make these scores incomparable with everyone
else's.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index as harrell_concordance

__all__ = [
    "censoring_distribution",
    "harrell_concordance",
    "antolini_concordance",
    "ipcw_brier",
    "calibration_table",
    "evaluate",
]

CensoringFn = Callable[[float], float]


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------
def _unpack(prediction: Any, times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Accept a ``SurvivalPrediction`` or a raw array plus explicit ``times``.

    Discriminated on ``.times``, not ``.mean`` — every numpy array has a ``mean``
    attribute, so testing that would misclassify a plain array.
    """
    if hasattr(prediction, "times"):
        return (np.asarray(prediction.mean, dtype=float),
                np.asarray(prediction.times, dtype=float))
    if times is None:
        raise ValueError(
            "times= is required when prediction is a plain array; pass the grid the "
            "survival function was evaluated on, or hand over the SurvivalPrediction."
        )
    return np.asarray(prediction, dtype=float), np.asarray(times, dtype=float)


def _check(surv: np.ndarray, time_grid: np.ndarray, t: np.ndarray) -> None:
    if surv.shape != (len(t), len(time_grid)):
        raise ValueError(
            f"surv has shape {surv.shape}, expected {(len(t), len(time_grid))} "
            "— (n_obs, n_times), matching t and time_grid."
        )
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing.")


def _surv_at(surv: np.ndarray, time_grid: np.ndarray, horizon: float) -> np.ndarray:
    """S(horizon | x) per row, linearly interpolated on the grid.

    Exact when ``horizon`` is a grid point, which is the usual case and the
    reason to build the grid with the horizons already in it.
    """
    horizon = float(horizon)
    if horizon < time_grid[0] or horizon > time_grid[-1]:
        raise ValueError(
            f"horizon {horizon:g} lies outside the prediction grid "
            f"[{time_grid[0]:g}, {time_grid[-1]:g}] — extend the grid rather than "
            "extrapolating a survival curve."
        )
    j = int(np.searchsorted(time_grid, horizon, side="left"))
    if time_grid[j] == horizon:
        return surv[:, j]
    lo, hi = j - 1, j
    w = (horizon - time_grid[lo]) / (time_grid[hi] - time_grid[lo])
    return surv[:, lo] * (1.0 - w) + surv[:, hi] * w


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------
def censoring_distribution(t: np.ndarray, event: np.ndarray) -> CensoringFn:
    """Return Ghat(u) = P(not yet censored past u), for IPCW weighting.

    Estimated by treating *censoring* as the event. Clipped away from zero so a
    weight can never explode in the tail where few subjects remain.

    Must be estimated on the **training** fold; ``evaluate`` enforces that by
    requiring the training data explicitly rather than defaulting to the fold
    being scored.
    """
    kmf = KaplanMeierFitter().fit(np.asarray(t, dtype=float),
                                  1 - np.asarray(event, dtype=int))

    def G(u: float) -> float:
        return float(np.clip(float(kmf.predict(u)), 1e-3, None))

    return G


def antolini_concordance(
    surv: np.ndarray,
    time_grid: np.ndarray,
    t: np.ndarray,
    event: np.ndarray,
) -> float:
    """Antolini's time-dependent concordance C-td.

    This is the metric reported by the published SUPPORT benchmarks (Kvamme,
    Borgan & Scheel 2019, JMLR 20(129)); Harrell's C is a different quantity and
    the two are not interchangeable.

    Harrell's C ranks subjects by a single scalar risk score. Antolini's compares
    each pair at the *earlier* subject's event time, so it respects predicted
    survival curves that cross — which is the whole point of a time-dependent
    measure::

        C-td = P( S_i(T_i) < S_j(T_i) | T_i < T_j, event_i = 1 )

    Ties in predicted survival count as half, which is what makes a
    covariate-independent prediction score exactly 0.5.

    Both members of a pair are read from the *same* grid column, so a step lookup
    is correct here and no interpolation is needed — unlike :func:`ipcw_brier`,
    which needs the value at one exact time.
    """
    surv = np.asarray(surv, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    t = np.asarray(t, dtype=float)
    event = np.asarray(event, dtype=int)
    _check(surv, time_grid, t)

    # column holding S at each subject's own observed time
    col = np.clip(np.searchsorted(time_grid, t, side="right") - 1, 0, len(time_grid) - 1)

    num = 0.0
    den = 0
    for i in np.flatnonzero(event == 1):
        later = t > t[i]              # strict: tied event times are not comparable
        if not later.any():
            continue
        s_i = surv[i, col[i]]
        s_j = surv[later, col[i]]
        num += float((s_i < s_j).sum()) + 0.5 * float((s_i == s_j).sum())
        den += int(later.sum())

    return num / den if den else float("nan")


def ipcw_brier(
    surv: np.ndarray,
    time_grid: np.ndarray,
    horizon: float,
    t: np.ndarray,
    event: np.ndarray,
    censoring: CensoringFn,
) -> float:
    """IPCW Brier score at one horizon — squared error on the predicted probability.

    Subjects censored before ``horizon`` have unknown status there. Dropping them
    biases the score and counting them as survivors is simply wrong, so the
    inverse-probability weights on the two classifiable groups carry their mass:

    =========================  =====================  =================
    subject                    contributes            weight
    =========================  =====================  =================
    failed before ``horizon``  ``(0 - S)**2``         ``1 / G(T_i)``
    known alive at ``horizon`` ``(1 - S)**2``         ``1 / G(horizon)``
    censored before it         nothing directly       carried by the above
    =========================  =====================  =================

    Lower is better. Unlike concordance this is a *proper* scoring rule, so it
    penalises miscalibration rather than only bad ranking — but it has a strong
    base-rate floor, so always read it against a covariate-free Kaplan-Meier
    baseline rather than against zero.
    """
    surv = np.asarray(surv, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    t = np.asarray(t, dtype=float)
    event = np.asarray(event, dtype=int)
    _check(surv, time_grid, t)

    S = _surv_at(surv, time_grid, horizon)
    died = (t <= horizon) & (event == 1)
    alive = t > horizon
    g_ti = np.array([censoring(x) for x in t])
    g_h = censoring(horizon)
    term = np.where(
        died,
        (0.0 - S) ** 2 / g_ti,
        np.where(alive, (1.0 - S) ** 2 / g_h, 0.0),
    )
    return float(term.mean())


def calibration_table(
    surv: np.ndarray,
    time_grid: np.ndarray,
    t: np.ndarray,
    event: np.ndarray,
    horizon: float,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Predicted vs observed event probability at ``horizon``, by predicted-risk bin.

    The observed side comes from a Kaplan-Meier fit *within each bin* rather than
    a raw proportion, for the same reason the Brier score needs IPCW: a raw
    proportion miscounts subjects censored before the horizon.

    Monotone and near-diagonal is the target. A systematic offset is
    miscalibration, which concordance is structurally blind to.
    """
    surv = np.asarray(surv, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    t = np.asarray(t, dtype=float)
    event = np.asarray(event, dtype=int)
    _check(surv, time_grid, t)

    risk = 1.0 - _surv_at(surv, time_grid, horizon)
    q = pd.qcut(risk, n_bins, labels=False, duplicates="drop")
    rows = []
    for b in np.unique(q):
        m = q == b
        k = KaplanMeierFitter().fit(t[m], event[m])
        rows.append({
            "bin": int(b) + 1,
            "n": int(m.sum()),
            "predicted": float(risk[m].mean()),
            "observed_KM": float(1 - k.predict(horizon)),
        })
    return pd.DataFrame(rows).set_index("bin")


def evaluate(
    prediction: Any,
    t: np.ndarray,
    event: np.ndarray,
    *,
    horizons: Sequence[float],
    train: tuple[np.ndarray, np.ndarray],
    times: np.ndarray | None = None,
    name: str | None = None,
    score_at: float | None = None,
    unit: str = "d",
) -> dict:
    """Score one model on a held-out fold; returns a row ready for ``pd.DataFrame``.

    Parameters
    ----------
    prediction
        A :class:`SurvivalPrediction` (its ``.times`` and ``.mean`` are used) or a
        raw ``(n_obs, n_times)`` array, in which case pass ``times``.
    t, event
        Observed times and event indicators for the fold being scored.
    horizons
        Times at which to report the Brier score.
    train
        ``(t_train, event_train)``. Required, and deliberately has no default: the
        censoring distribution must be estimated on the training fold, and letting
        it default to the fold being scored would silently produce a subtly wrong
        Brier score with nothing to indicate it.
    score_at
        Time at which predicted survival is read to form the risk score for
        Harrell's C. Defaults to the middle horizon.

    Returns
    -------
    dict
        ``{"model": name, "C-index": ..., "Antolini C-td": ..., "Brier@30d": ...}``.
        The ``model`` key is omitted when ``name`` is None. Keys are the column
        headers the row will carry into a DataFrame.
    """
    surv, grid = _unpack(prediction, times)
    t = np.asarray(t, dtype=float)
    event = np.asarray(event, dtype=int)
    _check(surv, grid, t)

    horizons = np.atleast_1d(np.asarray(horizons, dtype=float))
    t_train, e_train = train
    G = censoring_distribution(t_train, e_train)

    score_time = float(horizons[len(horizons) // 2]) if score_at is None else float(score_at)
    # higher predicted survival = later event, which is the ordering concordance_index wants
    score = _surv_at(surv, grid, score_time)

    row: dict[str, Any] = {}
    if name is not None:
        row["model"] = name
    row["C-index"] = float(harrell_concordance(t, score, event))
    row["Antolini C-td"] = antolini_concordance(surv, grid, t, event)
    for h in horizons:
        row[f"Brier@{h:g}{unit}"] = ipcw_brier(surv, grid, float(h), t, event, G)
    return row
