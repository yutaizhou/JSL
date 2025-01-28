# Example of a 1D pendulum problem applied to the Extended Kalman Filter,
# the Unscented Kalman Filter, and the Particle Filter (boostrap filter)
# Additionally, we test the particle filter when the observations have a 40%
# probability of being perturbed by a uniform(-2, 2) distribution

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jax.ops import index_update

import jsl.nlds.bootstrap_filter as b_lib
import jsl.nlds.extended_kalman_filter as ekf_lib
import jsl.nlds.unscented_kalman_filter as ukf_lib
from jsl.nlds.base import NLDS


def plot_filter_true(ax, time, estimate, obs, ground_truth, label, colors="tab:blue"):
    ax.plot(time, estimate, c="black", label=label)
    ax.scatter(time, obs, s=10, c="none", edgecolors=colors)
    ax.plot(time, ground_truth, c="gray", linewidth=8, alpha=0.5, label="true angle")
    ax.legend()


def fz(x, g=0.1, dt=0.1):
    x1, x2 = x[0], x[1]
    x1_new = x1 + x2 * dt
    x2_new = x2 - g * jnp.sin(x1) * dt
    return jnp.asarray([x1_new, x2_new])


def fx(x, *args):
    return jnp.sin(jnp.asarray([x[0]]))


def main():
    # *** Define initial configuration ***
    g = 10
    dt = 0.015
    qc = 0.06
    Q = jnp.array([[qc * dt**3 / 3, qc * dt**2 / 2], [qc * dt**2 / 2, qc * dt]])

    fx_vmap = jax.vmap(fx)
    fz_vec = jax.vmap(lambda x: fz(x, g=g, dt=dt))

    nsteps = 200
    Rt = jnp.eye(1) * 0.02
    x0 = jnp.array([1.5, 0.0]).astype(float)
    time = jnp.arange(0, nsteps * dt, dt)

    key = random.PRNGKey(3141)
    key_samples, key_pf, key_noisy = random.split(key, 3)
    model = NLDS(lambda x: fz(x, g=g, dt=dt), fx, Q, Rt)
    sample_state, sample_obs = model.sample(key, x0, nsteps)

    # *** Pertubed data ***
    key_noisy, key_values = random.split(key_noisy)
    sample_obs_noise = sample_obs.copy()
    samples_map = random.bernoulli(key_noisy, 0.5, (nsteps,))
    replacement_values = random.uniform(
        key_values, (samples_map.sum(),), minval=-2, maxval=2
    )
    sample_obs_noise = index_update(
        sample_obs_noise.ravel(), samples_map, replacement_values
    )
    colors = ["tab:red" if samp else "tab:blue" for samp in samples_map]

    # *** Perform filtering ****
    alpha, beta, kappa = 1, 0, 2
    state_size = 2
    Vinit = jnp.eye(state_size)
    ukf = NLDS(lambda x: fz(x, g=g, dt=dt), fx, Q, Rt, alpha, beta, kappa, state_size)
    particle_filter = NLDS(fz_vec, fx_vmap, Q, Rt)

    print("Filtering data...")
    _, ekf_hist = ekf_lib.filter(model, x0, sample_obs, return_params=["mean", "cov"])
    ekf_mean_hist, ekf_Sigma_hist = ekf_hist["mean"], ekf_hist["cov"]
    ukf_mean_hist, ukf_Sigma_hist = ukf_lib.filter(ukf, x0, sample_obs)
    pf_mean_hist = b_lib.filter(
        particle_filter, key_pf, x0, sample_obs, nsamples=4_000, Vinit=Vinit
    )

    print("Filtering outlier data...")
    _, ekf_perturbed_hist = ekf_lib.filter(
        model, x0, sample_obs_noise, return_params=["mean", "cov"]
    )
    ekf_perturbed_mean_hist, ekf_Sigma_hist = (
        ekf_perturbed_hist["mean"],
        ekf_perturbed_hist["cov"],
    )
    ukf_perturbed_mean_hist, ukf_Sigma_hist = ukf_lib.filter(ukf, x0, sample_obs_noise)
    pf_perturbed_mean_hist = b_lib.filter(
        particle_filter, key_pf, x0, sample_obs_noise, nsamples=2_000
    )

    ekf_estimate = fx_vmap(ekf_mean_hist)
    ukf_estimate = fx_vmap(ukf_mean_hist)
    pf_estimate = fx_vmap(pf_mean_hist)

    ekf_perturbed_estimate = fx_vmap(ekf_perturbed_mean_hist)
    ukf_perturbed_estimate = fx_vmap(ukf_perturbed_mean_hist)
    pf_perturbed_estimate = fx_vmap(pf_perturbed_mean_hist)
    ground_truth = fx_vmap(sample_state)

    dict_figures = {}
    # *** Plot results ***
    fig, ax = plt.subplots()
    plot_filter_true(ax, time, ekf_estimate, sample_obs, ground_truth, "Extended KF")
    dict_figures["pendulum_ekf_1d_demo"] = fig

    fig, ax = plt.subplots()
    plot_filter_true(ax, time, ukf_estimate, sample_obs, ground_truth, "Unscented KF")
    dict_figures["pendulum_ukf_1d_demo"] = fig

    fig, ax = plt.subplots()
    plot_filter_true(ax, time, pf_estimate, sample_obs, ground_truth, "Bootstrap PF")
    dict_figures["pendulum_pf_1d_demo"] = fig

    fig, ax = plt.subplots()
    plot_filter_true(
        ax,
        time,
        pf_perturbed_estimate,
        sample_obs_noise,
        ground_truth,
        "Bootstrap PF (noisy)",
        colors=colors,
    )
    dict_figures["pendulum_pf_noisy_1d_demo"] = fig

    fig, ax = plt.subplots()
    plot_filter_true(
        ax,
        time,
        ekf_perturbed_estimate,
        sample_obs_noise,
        ground_truth,
        "Extended KF (noisy)",
        colors=colors,
    )
    dict_figures["pendulum_ekf_noisy_1d_demo"] = fig

    fig, ax = plt.subplots()
    plot_filter_true(
        ax,
        time,
        ukf_perturbed_estimate,
        sample_obs_noise,
        ground_truth,
        "Unscented KF (noisy)",
        colors=colors,
    )
    dict_figures["pendulum_ukf_noisy_1d_demo"] = fig

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    figures = main()
    savefig(figures)
    plt.show()
