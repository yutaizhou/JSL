# Kalman filter agent
from typing import Callable, List

import chex
import jax.numpy as jnp

from jsl.nlds.extended_kalman_filter import ExtendedKalmanFilter
from jsl.sent.agents.agent import Agent


class EEKF(Agent):
    def __init__(
        self,
        fz: Callable,
        fx: Callable,
        Pt: chex.Array,
        Rt: Callable,
        mu: chex.Array,
        P0: chex.Array,
        return_params: List[str] = ["mean", "cov"],
    ):
        self.fz = fz
        self.fx = fx
        self.Pt = Pt
        self.Rt = Rt
        self.return_params = return_params

        self.prior_mean = mu
        self.prior_cov = P0

        self.reset(None)

    def update(self, X: chex.Array, y: chex.Array):
        (self.mu, self.Sigma), params = self.eekf.filter(
            self.mu,
            y,
            observations=X,
            Vinit=self.prior_cov,
            return_params=self.return_params,
        )
        return params

    def predict(self, x: chex.Array):
        return x @ self.mu.reshape((-1, 1))

    def reset(self, key: chex.Array):
        self.eekf = ExtendedKalmanFilter(self.fz, self.fx, self.Pt, self.Rt)
        self.mu = self.prior_mean
        self.Sigma = self.prior_cov

    def update(self, X: chex.Array, y: chex.Array):
        (self.mu, self.Sigma), params = self.eekf.filter(
            self.mu,
            y,
            observations=X,
            Vinit=self.prior_cov,
            return_params=self.return_params,
        )
        return params

    def predict(self, x: chex.Array):
        return x @ self.mu.reshape((-1, 1))

    def reset(self, key: chex.Array):
        self.eekf = ExtendedKalmanFilter(self.fz, self.fx, self.Pt, self.Rt)
        self.mu = self.prior_mean
        self.Sigma = self.prior_cov
