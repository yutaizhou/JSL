# Online Bayesian linear regression in 1d using Kalman Filter
# Based on: https://github.com/probml/pmtk3/blob/master/demos/linregOnlineDemoKalman.m

# The latent state corresponds to the current estimate of the regression weights w.
# The observation model has the form
# p(y(t) |  w(t), x(t)) = Gauss( C(t) * w(t), R(t))
# where C(t) = X(t,:) is the observation matrix for step t.
# The dynamics model has the form
# p(w(t) | w(t-1)) = Gauss(A * w(t-1), Q)
# where Q>0 allows for parameter  drift.
# We show that the result is equivalent to batch (offline) Bayesian inference.

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import scan
from numpy.linalg import inv

from jsl.lds.kalman_filter import KalmanFilter


def kf_linreg(X, y, R, mu0, Sigma0, F, Q):
    """
    Online estimation of a linear regression
    using Kalman Filters
    Parameters
    ----------
    X: array(n_obs, dimension)
        Matrix of features
    y: array(n_obs,)
        Array of observations
    Q: float
        Known variance
    mu0: array(dimension)
        Prior mean
    Sigma0: array(dimesion, dimension)
        Prior covariance matrix
    Returns
    -------
    * array(n_obs, dimension)
        Online estimation of parameters
    * array(n_obs, dimension, dimension)
        Online estimation of uncertainty
    """
    n_obs, dim = X.shape
    C = lambda t: X[t][None, ...]
    kf = KalmanFilter(F, C, Q, R, mu0.copy(), Sigma0.copy(), timesteps=n_obs)
    _, (mu_hist, Sigma_hist, _, _) = scan(
        kf.kalman_step, (mu0.copy(), Sigma0.copy(), 0), y
    )
    return mu_hist, Sigma_hist


def posterior_lreg(X, y, R, mu0, Sigma0):
    """
    Compute mean and covariance matrix of a
    Bayesian Linear regression
    Parameters
    ----------
    X: array(n_obs, dimension)
        Matrix of features
    y: array(n_obs,)
        Array of observations
    R: float
        Known variance
    mu0: array(dimension)
        Prior mean
    Sigma0: array(dimesion, dimension)
        Prior covariance matrix
    Returns
    -------
    * array(dimension)
        Posterior mean
    * array(n_obs, dimension, dimension)
        Posterior covariance matrix
    """
    Sn_bayes_inv = inv(Sigma0) + X.T @ X / R
    Sn_bayes = inv(Sn_bayes_inv)
    mn_bayes = Sn_bayes @ (inv(Sigma0) @ mu0 + X.T @ y / R)

    return mn_bayes, Sn_bayes


def main():
    n_obs = 21
    timesteps = jnp.arange(n_obs)
    x = jnp.linspace(0, 20, n_obs)
    X = jnp.c_[jnp.ones(n_obs), x]
    F = jnp.eye(2)
    mu0 = jnp.zeros(2)
    Sigma0 = jnp.eye(2) * 10.0

    Q, R = 0, 1
    # Data from original matlab example
    y = jnp.array(
        [
            2.4865,
            -0.3033,
            -4.0531,
            -4.3359,
            -6.1742,
            -5.604,
            -3.5069,
            -2.3257,
            -4.6377,
            -0.2327,
            -1.9858,
            1.0284,
            -2.264,
            -0.4508,
            1.1672,
            6.6524,
            4.1452,
            5.2677,
            6.3403,
            9.6264,
            14.7842,
        ]
    )

    # Online estimation
    mu_hist, Sigma_hist = kf_linreg(X, y, R, mu0, Sigma0, F, Q)
    kf_var = Sigma_hist[-1, [0, 1], [0, 1]]
    w0_hist, w1_hist = mu_hist.T
    w0_err, w1_err = jnp.sqrt(Sigma_hist[:, [0, 1], [0, 1]].T)

    # Offline estimation
    (w0_post, w1_post), Sigma_post = posterior_lreg(X, y, R, mu0, Sigma0)
    w0_std, w1_std = jnp.sqrt(Sigma_post[[0, 1], [0, 1]])

    dict_figures = {}

    fig, ax = plt.subplots()
    ax.errorbar(
        timesteps,
        w0_hist,
        w0_err,
        fmt="-o",
        label="$w_0$",
        color="black",
        fillstyle="none",
    )
    ax.errorbar(timesteps, w1_hist, w1_err, fmt="-o", label="$w_1$", color="tab:red")

    ax.axhline(y=w0_post, c="black", label="$w_0$ batch")
    ax.axhline(y=w1_post, c="tab:red", linestyle="--", label="$w_1$ batch")

    ax.fill_between(
        timesteps, w0_post - w0_std, w0_post + w0_std, color="black", alpha=0.4
    )
    ax.fill_between(
        timesteps, w1_post - w1_std, w1_post + w1_std, color="tab:red", alpha=0.4
    )

    plt.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("weights")
    ax.set_ylim(-8, 4)
    ax.set_xlim(-0.5, n_obs)
    dict_figures["linreg_online_kalman"] = fig

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    dict_figures = main()
    savefig(dict_figures)
    plt.show()
