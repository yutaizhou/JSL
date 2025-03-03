# This script produces an illustration of Kalman filtering and smoothing

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

from jsl.demos.plot_utils import plot_ellipse
from jsl.lds.kalman_filter import LDS, filter, smooth


def plot_tracking_values(observed, filtered, cov_hist, signal_label, ax):
    """
    observed: array(nsteps, 2)
        Array of observed values
    filtered: array(nsteps, state_size)
        Array of latent (hidden) values. We consider only the first
        two dimensions of the latent values
    cov_hist: array(nsteps, state_size, state_size)
        History of the retrieved (filtered) covariance matrices
    ax: matplotlib AxesSubplot
    """
    timesteps, _ = observed.shape
    ax.plot(
        observed[:, 0],
        observed[:, 1],
        marker="o",
        linewidth=0,
        markerfacecolor="none",
        markeredgewidth=2,
        markersize=8,
        label="observed",
        c="tab:green",
    )
    ax.plot(
        *filtered[:, :2].T, label=signal_label, c="tab:red", marker="x", linewidth=2
    )
    for t in range(0, timesteps, 1):
        covn = cov_hist[t][:2, :2]
        plot_ellipse(covn, filtered[t, :2], ax, n_std=2.0, plot_center=False)
    ax.axis("equal")
    ax.legend()


def sample_filter_smooth(key, lds_model, timesteps):
    """
    Sample from a linear dynamical system, apply the kalman filter
    (forward pass), and performs smoothing.

    Parameters
    ----------
    lds: LinearDynamicalSystem
        Instance of a linear dynamical system with known parameters

    Returns
    -------
    Dictionary with the following key, values
    * (z_hist) array(timesteps, state_size):
        Simulation of Latent states
    * (x_hist) array(timesteps, observation_size):
        Simulation of observed states
    * (mu_hist) array(timesteps, state_size):
        Filtered means mut
    * (Sigma_hist) array(timesteps, state_size, state_size)
        Filtered covariances Sigmat
    * (mu_cond_hist) array(timesteps, state_size)
        Filtered conditional means mut|t-1
    * (Sigma_cond_hist) array(timesteps, state_size, state_size)
        Filtered conditional covariances Sigmat|t-1
    * (mu_hist_smooth) array(timesteps, state_size):
        Smoothed means mut
    * (Sigma_hist_smooth) array(timesteps, state_size, state_size)
        Smoothed covariances Sigmat
    """
    z_hist, x_hist = lds_model.sample(key, timesteps)
    mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = filter(lds_model, x_hist)
    mu_hist_smooth, Sigma_hist_smooth = smooth(
        lds_model, mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist
    )

    return {
        "z_hist": z_hist,
        "x_hist": x_hist,
        "mu_hist": mu_hist,
        "Sigma_hist": Sigma_hist,
        "mu_cond_hist": mu_cond_hist,
        "Sigma_cond_hist": Sigma_cond_hist,
        "mu_hist_smooth": mu_hist_smooth,
        "Sigma_hist_smooth": Sigma_hist_smooth,
    }


def main():
    key = random.PRNGKey(314)
    timesteps = 15
    delta = 1.0
    A = jnp.array([[1, 0, delta, 0], [0, 1, 0, delta], [0, 0, 1, 0], [0, 0, 0, 1]])

    C = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    state_size, _ = A.shape
    observation_size, _ = C.shape

    Q = jnp.eye(state_size) * 0.001
    R = jnp.eye(observation_size) * 1.0
    # Prior parameter distribution
    mu0 = jnp.array([8, 10, 1, 0]).astype(float)
    Sigma0 = jnp.eye(state_size) * 1.0

    lds_instance = LDS(A, C, Q, R, mu0, Sigma0)
    result = sample_filter_smooth(key, lds_instance, timesteps)

    l2_filter = jnp.linalg.norm(result["z_hist"][:, :2] - result["mu_hist"][:, :2], 2)
    l2_smooth = jnp.linalg.norm(
        result["z_hist"][:, :2] - result["mu_hist_smooth"][:, :2], 2
    )

    print(f"L2-filter: {l2_filter:0.4f}")
    print(f"L2-smooth: {l2_smooth:0.4f}")

    dict_figures = {}

    fig_truth, axs = plt.subplots()
    axs.plot(
        result["x_hist"][:, 0],
        result["x_hist"][:, 1],
        marker="o",
        linewidth=0,
        markerfacecolor="none",
        markeredgewidth=2,
        markersize=8,
        label="observed",
        c="tab:green",
    )

    axs.plot(
        result["z_hist"][:, 0],
        result["z_hist"][:, 1],
        linewidth=2,
        label="truth",
        marker="s",
        markersize=8,
    )
    axs.legend()
    axs.axis("equal")
    dict_figures["kalman-tracking-truth"] = fig_truth

    fig_filtered, axs = plt.subplots()
    plot_tracking_values(
        result["x_hist"], result["mu_hist"], result["Sigma_hist"], "filtered", axs
    )
    dict_figures["kalman-tracking-filtered"] = fig_filtered

    fig_smoothed, axs = plt.subplots()
    plot_tracking_values(
        result["x_hist"],
        result["mu_hist_smooth"],
        result["Sigma_hist_smooth"],
        "smoothed",
        axs,
    )
    dict_figures["kalman-tracking-smoothed"] = fig_smoothed

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    figures = main()
    savefig(figures)
    plt.show()
