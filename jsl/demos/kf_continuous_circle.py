# Example of a Kalman Filter procedure
# on a continuous system with imaginary eigenvalues
# and discrete samples
# For futher reference and examples see:
#   * Section on Kalman Filters in PML vol2 book
#   * Nonlinear Dynamics and Chaos - Steven Strogatz

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from jsl.demos.plot_utils import plot_ellipse
from jsl.lds.cont_kalman_filter import ContinuousKalmanFilter


def main():
    A = jnp.array([[0, 1], [-1, 0]])
    C = jnp.eye(2)

    dt = 0.01
    T = 5.5
    nsamples = 70
    x0 = jnp.array([0.5, -0.75])

    # State noise
    Qt = jnp.eye(2) * 0.001
    # Observed noise
    Rt = jnp.eye(2) * 0.01

    Sigma0 = jnp.eye(2)

    key = random.PRNGKey(314)
    kf = ContinuousKalmanFilter(A, C, Qt, Rt, x0, Sigma0)
    sample_state, sample_obs, jump = kf.sample(key, x0, T, nsamples)
    mu_hist, V_hist, *_ = kf.filter(sample_obs, jump, dt)

    step = 0.1
    vmin, vmax = -1.5, 1.5 + step
    X = np.mgrid[-1:1.5:step, vmin:vmax:step][::-1]
    X_dot = jnp.einsum("ij,jxy->ixy", A, X)

    dict_figures = {}

    fig_state, ax = plt.subplots()
    ax.plot(*sample_state.T, label="state space")
    ax.scatter(*sample_obs.T, marker="+", c="tab:green", s=60, label="observations")
    ax.scatter(*sample_state[0], c="black", zorder=3)
    field = ax.streamplot(*X, *X_dot, density=1.1, color="#ccccccaa")
    ax.legend()
    plt.axis("equal")
    ax.set_title("State Space")
    dict_figures["kf-circle-state"] = fig_state

    fig_filtered, ax = plt.subplots()
    ax.plot(*mu_hist.T, c="tab:orange", label="Filtered")
    ax.scatter(*sample_obs.T, marker="+", s=60, c="tab:green", label="observations")
    ax.scatter(*mu_hist[0], c="black", zorder=3)
    for mut, Vt in zip(mu_hist[::4], V_hist[::4]):
        plot_ellipse(Vt, mut, ax, plot_center=False, alpha=0.9, zorder=3)
    plt.legend()
    field = ax.streamplot(*X, *X_dot, density=1.1, color="#ccccccaa")
    ax.legend()
    plt.axis("equal")
    ax.set_title("Approximate Space")
    dict_figures["kf-circle-filtered"] = fig_filtered

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    dict_figures = main()
    savefig(dict_figures)
    plt.show()
