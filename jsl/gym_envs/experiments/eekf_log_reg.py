import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jax.nn import sigmoid

# Import data and baseline solution
from jsl.demos import logreg_biclusters as demo
from jsl.gym_envs.configs.classification_eekf_test import get_config
from jsl.gym_envs.run import main as train


def main():
    figures, data = demo.main()
    X = data["X"]
    y = data["y"]
    Phi = data["Phi"]
    Xspace = data["Xspace"]
    Phispace = data["Phispace"]
    w_laplace = data["w_laplace"]

    n_datapoints, M = Phi.shape
    colors = ["black" if el else "white" for el in y]

    # Predictive domain
    xmin, ymin = X.min(axis=0) - 0.1
    xmax, ymax = X.max(axis=0) + 0.1
    step = 0.1
    Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
    _, nx, ny = Xspace.shape
    Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])

    config = get_config()
    hist, _ = train(config)

    w_eekf_hist = hist["mean"]
    P_eekf_hist = hist["cov"]

    w_eekf = hist["mean"][-1]
    P_eekf = hist["mean"][-1]

    ### *** Ploting surface predictive distribution ***
    colors = ["black" if el else "white" for el in y]
    dict_figures = {}
    key = random.PRNGKey(31415)
    nsamples = 5000

    # EEKF surface predictive distribution
    eekf_samples = random.multivariate_normal(key, w_eekf, P_eekf, (nsamples,))
    Z_eekf = sigmoid(jnp.einsum("mij,sm->sij", Phispace, eekf_samples))
    Z_eekf = Z_eekf.mean(axis=0)

    fig_eekf, ax = plt.subplots()
    title = "EEKF  Predictive Distribution"
    demo.plot_posterior_predictive(ax, X, Xspace, Z_eekf, title, colors)
    dict_figures["logistic_regression_surface_eekf"] = fig_eekf
    plt.savefig("logistic_regression_surface_eekf.png")
    ### Plot EEKF and Laplace training history
    P_eekf_hist_diag = jnp.diagonal(P_eekf_hist, axis1=1, axis2=2)
    # P_laplace_diag = jnp.sqrt(jnp.diagonal(SN))
    lcolors = ["black", "tab:blue", "tab:red"]
    elements = w_eekf_hist.T, P_eekf_hist_diag.T, w_laplace, lcolors
    timesteps = jnp.arange(n_datapoints) + 1

    for k, (wk, Pk, wk_laplace, c) in enumerate(zip(*elements)):
        fig_weight_k, ax = plt.subplots()
        ax.errorbar(timesteps, wk, jnp.sqrt(Pk), c=c, label=f"$w_{k}$ online (EEKF)")
        ax.axhline(
            y=wk_laplace,
            c=c,
            linestyle="dotted",
            label=f"$w_{k}$ batch (Laplace)",
            linewidth=3,
        )

        ax.set_xlim(1, n_datapoints)
        ax.legend(framealpha=0.7, loc="upper right")
        ax.set_xlabel("number samples")
        ax.set_ylabel("weights")
        plt.tight_layout()
        dict_figures[f"logistic_regression_hist_ekf_w{k}"] = fig_weight_k
        plt.savefig(f"logistic_regression_hist_ekf_w{k}.png")

    print("EEKF weights")
    print(w_eekf, end="\n" * 2)

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    figs = main()
    savefig(figs)
    plt.show()
