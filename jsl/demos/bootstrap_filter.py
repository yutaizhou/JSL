# Demo of the bootstrap filter under a
# nonlinear discrete system

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

from jsl.nlds.base import NLDS
from jsl.nlds.bootstrap_filter import filter


def plot_samples(sample_state, sample_obs, ax=None):
    fig, ax = plt.subplots()
    ax.plot(*sample_state.T, label="state space")
    ax.scatter(*sample_obs.T, s=60, c="tab:green", marker="+")
    ax.scatter(*sample_state[0], c="black", zorder=3)
    ax.legend()
    ax.set_title("Noisy observations from hidden trajectory")
    plt.axis("equal")
    return fig


def plot_inference(sample_obs, mean_hist):
    fig, ax = plt.subplots()
    ax.scatter(*sample_obs.T, marker="+", color="tab:green", s=60)
    ax.plot(*mean_hist.T, c="tab:orange", label="filtered")
    ax.scatter(*mean_hist[0], c="black", zorder=3)
    plt.legend()
    plt.axis("equal")
    return fig


def main():
    def fz(x, dt):
        return x + dt * jnp.array([jnp.sin(x[1]), jnp.cos(x[0])])

    def fx(x):
        return x

    dt = 0.4
    nsteps = 100
    # Initial state vector
    x0 = jnp.array([1.5, 0.0])
    # State noise
    Qt = jnp.eye(2) * 0.001
    # Observed noise
    Rt = jnp.eye(2) * 0.05

    key = random.PRNGKey(314)
    model = NLDS(lambda x: fz(x, dt), fx, Qt, Rt)
    sample_state, sample_obs = model.sample(key, x0, nsteps)

    n_particles = 3_000
    fz_vec = jax.vmap(fz, in_axes=(0, None))
    particle_filter = NLDS(lambda x: fz_vec(x, dt), fx, Qt, Rt)
    pf_mean = filter(particle_filter, key, x0, sample_obs, n_particles)

    dict_figures = {}
    fig_boostrap = plot_inference(sample_obs, pf_mean)
    dict_figures["nlds2d_bootstrap"] = fig_boostrap

    fig_data = plot_samples(sample_state, sample_obs)
    dict_figures["nlds2d_data"] = fig_data

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    dict_figures = main()
    savefig(dict_figures)
    plt.show()
