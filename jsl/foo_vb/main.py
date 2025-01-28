"""
As stated in the original paper Task Agnostic Continual Learning Using Online Variational Bayes with Fixed-Point Updates,

Foo-vb is the novel fixed-point equations for the online variational Bayes optimization problem,
 for multivariate Gaussian parametric distributions.

The original FOO-VB Pytorch implementation is available at https://github.com/chenzeno/FOO-VB.
This library is Jax implementation based on the original code.

Author: Aleyna Kara(@karalleyna)

"""

from jax.config import config

config.update("jax_enable_x64", True)

from jax.config import config

config.update("jax_debug_nans", True)
from random import randint
from typing import Callable, Sequence

import datasets as ds
import flax.linen as nn
import foo_vb_lib
import ml_collections
import run
from jax import random

"""from jax.config import config
config.update("jax_enable_x64", True)"""


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.batch_size = 128
    config.test_batch_size = 1000

    config.epochs = 20
    config.seed = 1
    config.train_mc_iters = 3

    config.s_init = 0.27
    config.eta = 1.0
    config.alpha = 0.5

    config.tasks = 10
    config.results_dir = "."

    config.dataset = "continuous_permuted_mnist"
    config.iterations_per_virtual_epc = 468

    config.diagonal = True

    return config


class Net(nn.Module):
    features: Sequence[int]
    activation_fn: Callable = nn.activation.relu

    @nn.compact
    def __call__(self, x):
        for feature in self.features[:-1]:
            x = self.activation_fn(nn.Dense(feature)(x))
        return nn.log_softmax(nn.Dense(self.features[-1])(x))


Net100 = Net([100, 100, 10])
Net200 = Net([200, 200, 10])

if __name__ == "__main__":
    config = get_config()
    config.alpha = 0.6
    model = Net200
    key = random.PRNGKey(0)
    perm_key, key = random.split(key)

    image_size = 784
    n_permutations = 10

    permutations = foo_vb_lib.create_random_perm(perm_key, image_size, n_permutations)
    permutations = permutations[1:11]
    train_loaders, test_loaders = ds.ds_padded_cont_permuted_mnist(
        num_epochs=int(config.epochs * config.tasks),
        iterations_per_virtual_epc=config.iterations_per_virtual_epc,
        contpermuted_beta=4,
        permutations=permutations,
        batch_size=config.batch_size,
    )

    ava_test = run.train_continuous_mnist(
        key, model, train_loaders, test_loaders, image_size, n_permutations, config
    )
    print(ava_test)
