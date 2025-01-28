import chex
import jax.numpy as jnp
from jax import nn, random


def classification_loss(logprobs, targets):
    nclasses = logprobs.shape[-1]
    one_hot_targets = nn.one_hot(targets, nclasses, axis=-1)
    nll = jnp.sum(logprobs * one_hot_targets, axis=-1)
    ce = -jnp.mean(nll)
    return ce


def regression_loss(predictions, outputs):
    return jnp.mean(jnp.power(predictions - outputs, 2))


class SequentialDataEnvironment:
    def __init__(
        self,
        X_train: chex.Array,
        y_train: chex.Array,
        X_test: chex.Array,
        y_test: chex.Array,
        train_batch_size: int,
        test_batch_size: int,
        classification: bool,
    ):
        ntrain, nfeatures = X_train.shape
        ntest, _ = X_test.shape
        _, out = y_train.shape

        # TODO: It will produce an error if ntrain % train_batch_size != 0
        ntrain_batches = ntrain // train_batch_size
        # TODO: It will produce an error if ntest % test_batch_size != 0
        ntest_batches = ntest // test_batch_size

        self.X_train = jnp.reshape(
            X_train, [ntrain_batches, train_batch_size, nfeatures]
        )
        self.y_train = jnp.reshape(y_train, [ntrain_batches, train_batch_size, out])

        self.X_test = jnp.reshape(X_test, [ntest_batches, test_batch_size, nfeatures])
        self.y_test = jnp.reshape(y_test, [ntest_batches, test_batch_size, out])

        if classification:
            self.loss_fn = classification_loss
        else:
            self.loss_fn = regression_loss

    def get_data(self, t: int):
        return self.X_train[t], self.y_train[t], self.X_test, self.y_test

    def reward(self, y_pred: chex.Array, y_test: chex.Array):
        loss = self.loss_fn(y_pred, y_test)
        return loss

    def shuffle_data(self, key: chex.PRNGKey):
        train_key, test_key = random.split(key)
        ntrain = len(self.X_train)
        train_indices = jnp.arange(ntrain)
        train_indices = random.shuffle(train_key, train_indices)

        self.X_train = self.X_train[train_indices]
        self.y_train = self.y_train[train_indices]

        ntest = len(self.X_test)
        test_indices = jnp.arange(ntest)
        test_indices = random.shuffle(test_key, test_indices)

        self.X_test = self.X_test[test_indices]
        self.y_test = self.y_test[test_indices]

    def reset(self, key: chex.PRNGKey):
        ntrain_batches, train_batch_size, nfeatures = self.X_train.shape
        ntest_batches, test_batch_size, out = self.y_test.shape

        self.shuffle_data(key)

        self.X_train = jnp.reshape(
            self.X_train, [ntrain_batches, train_batch_size, nfeatures]
        )
        self.y_train = jnp.reshape(
            self.y_train, [ntrain_batches, train_batch_size, out]
        )

        self.X_test = jnp.reshape(
            self.X_test, [ntest_batches, test_batch_size, nfeatures]
        )
        self.y_test = jnp.reshape(self.y_test, [ntest_batches, test_batch_size, out])
