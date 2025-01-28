# Extended Rauch-Tung-Striebel smoother or Extended Kalman Smoother (EKS)
from functools import partial
from typing import Callable, Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp

from jsl.nlds import extended_kalman_filter as ekf

from .base import NLDS


def smooth_step(
    state: Tuple[chex.Array, chex.Array, int],
    xs: Tuple[chex.Array, chex.Array],
    params: NLDS,
    Dfz: Callable,
    eps: float,
    return_params: Dict,
) -> Tuple[Tuple[chex.Array, chex.Array, int], Dict]:
    mean_next, cov_next, t = state
    mean_kf, cov_kf = xs

    mean_next_hat = params.fz(mean_kf)
    cov_next_hat = Dfz(mean_kf) @ cov_kf @ Dfz(mean_kf).T + params.Qz(mean_kf, t)
    cov_next_hat_eps = cov_next_hat + eps * jnp.eye(mean_next_hat.shape[0])
    kalman_gain = jnp.linalg.solve(cov_next_hat_eps, Dfz(mean_kf).T) @ cov_kf

    mean_prev = mean_kf + kalman_gain @ (mean_next - mean_next_hat)
    cov_prev = cov_kf + kalman_gain @ (cov_next - cov_next_hat) @ kalman_gain.T

    prev_state = (mean_prev, cov_prev, t - 1)
    carry = {"mean": mean_prev, "cov": cov_prev}
    carry = {key: val for key, val in carry.items() if key in return_params}

    return prev_state, carry


def smooth(
    params: NLDS,
    init_state: chex.Array,
    observations: chex.Array,
    covariates: chex.Array = None,
    Vinit: chex.Array = None,
    return_params: List = None,
    eps: float = 0.001,
    return_filter_history: bool = False,
) -> Dict[str, Dict[str, chex.Array]]:
    kf_params = ["mean", "cov"]
    Dfz = jax.jacrev(params.fz)
    _, hist_filter = ekf.filter(
        params,
        init_state,
        observations,
        covariates,
        Vinit,
        return_params=kf_params,
        eps=eps,
        return_history=True,
    )
    kf_hist_mean, kf_hist_cov = hist_filter["mean"], hist_filter["cov"]
    kf_last_mean, kf_hist_mean = kf_hist_mean[-1], kf_hist_mean[:-1]
    kf_last_cov, kf_hist_cov = kf_hist_cov[-1], kf_hist_cov[:-1]

    smooth_step_partial = partial(
        smooth_step, params=params, Dfz=Dfz, eps=eps, return_params=return_params
    )

    init_state = (kf_last_mean, kf_last_cov, len(kf_hist_mean) - 1)
    xs = (kf_hist_mean, kf_hist_cov)
    _, hist_smooth = jax.lax.scan(smooth_step_partial, init_state, xs, reverse=True)

    hist = {
        "smooth": hist_smooth,
        "filter": hist_filter if return_filter_history else None,
    }

    return hist
