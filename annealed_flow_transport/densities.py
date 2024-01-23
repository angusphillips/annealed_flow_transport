# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for probability densities."""

import abc
import pickle

from annealed_flow_transport import vae as vae_lib
import annealed_flow_transport.aft_types as tp
import annealed_flow_transport.cox_process_utils as cp_utils
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.scipy.linalg as slinalg
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jax.scipy.stats import norm
from jax.flatten_util import ravel_pytree
import numpy as np
import tensorflow_datasets as tfds
import numpyro

# TypeDefs
NpArray = np.ndarray
Array = tp.Array
ConfigDict = tp.ConfigDict
Samples = tp.Samples
SampleShape = tp.SampleShape

assert_trees_all_equal = chex.assert_trees_all_equal


def pad_with_const(x):
    extra = np.ones((x.shape[0], 1))
    return np.hstack([extra, x])


def standardize_and_pad(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std
    return pad_with_const(x)


def load_data(path: str):
    with open(path, mode="rb") as f:
        x, y = pickle.load(f)
    y = (y + 1) // 2
    x = standardize_and_pad(x)
    return x, y


class LogDensity(metaclass=abc.ABCMeta):
    """Abstract base class from which all log densities should inherit."""

    def __init__(
        self, config: ConfigDict, sample_shape: SampleShape, is_target: bool = False
    ):
        self._check_constructor_inputs(config, sample_shape)
        self._config = config
        self._sample_shape = sample_shape
        self._is_target = is_target
        self.dim = self._sample_shape[-1]

    @abc.abstractmethod
    def _check_constructor_inputs(
        self, config: ConfigDict, sample_shape: SampleShape, is_target: bool
    ):
        """Check the config and sample shape of the class.

        Will typically raise Assertion like errors.

        Args:
          config: Configuration for the log density.
          sample_shape: Shape expected for the density.
        """

    def __call__(self, x: Samples, density_state: int) -> tp.Tuple[Array, int]:
        """Evaluate the log density with automatic shape checking.

        This calls evaluate_log_density which needs to be implemented
        in derived classes.

        Args:
          x: input Samples.
        Returns:
          Array of shape (num_batch,) with corresponding log densities.
        """
        self._check_input_shape(x)
        output, density_state = self.evaluate_log_density(x, density_state)
        self._check_output_shape(x, output)
        return output, density_state

    @abc.abstractmethod
    def evaluate_log_density(
        self, x: Samples, density_state: int
    ) -> tp.Tuple[Array, int]:
        """Evaluate the log density.

        Args:
          x: Samples.
        Returns:
          Array of shape (num_batch,) containing values of log densities.
        """

    def _check_input_shape(self, x_in: Samples):
        should_be_tree_shape = jax.tree_map(lambda x: x.shape[1:], x_in)
        chex.assert_trees_all_equal(self._sample_shape, should_be_tree_shape)

        def get_first_leaf(tree):
            return jax.tree_util.tree_leaves(tree)[0]

        first_batch_size = np.shape(get_first_leaf(x_in))[0]
        chex.assert_tree_shape_prefix(x_in, (first_batch_size,))

    def _check_output_shape(self, x_in: Samples, x_out: Samples):
        batch_sizes = jax.tree_util.tree_map(lambda x: np.shape(x)[0], x_in)

        def get_first_leaf(tree):
            return jax.tree_util.tree_leaves(tree)[0]

        first_batch_size = get_first_leaf(batch_sizes)
        chex.assert_shape(x_out, (first_batch_size,))

    def _check_members_types(self, config: ConfigDict, expected_members_types):
        for elem, elem_type in expected_members_types:
            if elem not in config:
                raise ValueError("LogDensity config element not found: ", elem)
            if not isinstance(config[elem], elem_type):
                msg = (
                    "LogDensity config element "
                    + elem
                    + " is not of type "
                    + str(elem_type)
                )
                raise TypeError(msg)


class NormalDistribution(LogDensity):
    """A univariate normal distribution with configurable scale and location.

    num_dim should be 1 and config should include scalars "loc" and "scale"
    """

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        assert_trees_all_equal(sample_shape, (1,))
        expected_members_types = [
            ("loc", float),
            ("scale", float),
        ]

        self._check_members_types(config, expected_members_types)

    def evaluate_log_density(
        self, x: Samples, density_state: int
    ) -> tp.Tuple[Array, int]:
        output = norm.logpdf(x, loc=self._config.loc, scale=self._config.scale)[:, 0]
        density_state += self._is_target * x.shape[0]
        return output, density_state


class MultivariateNormalDistribution(LogDensity):
    """A normalized multivariate normal distribution.

    Each element of the mean vector has the same value config.shared_mean
    Each element of the diagonal covariance matrix has value config.diagonal_cov
    """

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        expected_members_types = [("shared_mean", float), ("diagonal_cov", float)]
        assert len(sample_shape) == 1
        self._check_members_types(config, expected_members_types)

    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        num_dim = np.shape(x)[1]
        mean = jnp.ones(num_dim) * self._config.shared_mean
        cov = jnp.diag(jnp.ones(num_dim) * self._config.diagonal_cov)
        output = multivariate_normal.logpdf(x, mean=mean, cov=cov)
        density_state += self._is_target * x.shape[0]
        return output, density_state


class FunnelDistribution(LogDensity):
    """The funnel distribution from https://arxiv.org/abs/physics/0009028.

    num_dim should be 10. config is unused in this case.
    """

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        del config
        assert_trees_all_equal(sample_shape, (10,))

    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        num_dim = self._sample_shape[0]

        def unbatched(x):
            v = x[0]
            log_density_v = norm.logpdf(v, loc=0.0, scale=3.0)
            variance_other = jnp.exp(v)
            other_dim = num_dim - 1
            cov_other = jnp.eye(other_dim) * variance_other
            mean_other = jnp.zeros(other_dim)
            log_density_other = multivariate_normal.logpdf(
                x[1:], mean=mean_other, cov=cov_other
            )
            chex.assert_equal_shape([log_density_v, log_density_other])
            return log_density_v + log_density_other

        output = jax.vmap(unbatched)(x)
        density_state += self._is_target * x.shape[0]
        return output, density_state


class LogGaussianCoxPines(LogDensity):
    """Log Gaussian Cox process posterior in 2D for pine saplings data.

    This follows Heng et al 2020 https://arxiv.org/abs/1708.08396 .

    config.file_path should point to a csv file of num_points columns
    and 2 rows containg the Finnish pines data.

    config.use_whitened is a boolean specifying whether or not to use a
    reparameterization in terms of the Cholesky decomposition of the prior.
    See Section G.4 of https://arxiv.org/abs/2102.07501 for more detail.
    The experiments in the paper have this set to False.

    num_dim should be the square of the lattice sites per dimension.
    So for a 40 x 40 grid num_dim should be 1600.
    """

    def __init__(
        self, config: ConfigDict, sample_shape: SampleShape, is_target: bool = False
    ):
        super().__init__(config, sample_shape, is_target)

        # Discretization is as in Controlled Sequential Monte Carlo
        # by Heng et al 2017 https://arxiv.org/abs/1708.08396
        num_dim = sample_shape[0]
        self._num_latents = num_dim
        self._num_grid_per_dim = int(np.sqrt(num_dim))

        bin_counts = jnp.array(
            cp_utils.get_bin_counts(
                self.get_pines_points(config.file_path), self._num_grid_per_dim
            )
        )

        self._flat_bin_counts = jnp.reshape(bin_counts, (self._num_latents))

        # This normalizes by the number of elements in the grid
        self._poisson_a = 1.0 / self._num_latents
        # Parameters for LGCP are as estimated in Moller et al, 1998
        # "Log Gaussian Cox processes" and are also used in Heng et al.

        self._signal_variance = 1.91
        self._beta = 1.0 / 33

        self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

        def short_kernel_func(x, y):
            return cp_utils.kernel_func(
                x, y, self._signal_variance, self._num_grid_per_dim, self._beta
            )

        self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
        self._cholesky_gram = jnp.linalg.cholesky(self._gram_matrix)
        self._white_gaussian_log_normalizer = (
            -0.5 * self._num_latents * jnp.log(2.0 * jnp.pi)
        )

        half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self._cholesky_gram))))
        self._unwhitened_gaussian_log_normalizer = (
            -0.5 * self._num_latents * jnp.log(2.0 * jnp.pi) - half_log_det_gram
        )
        # The mean function is a constant with value mu_zero.
        self._mu_zero = jnp.log(126.0) - 0.5 * self._signal_variance

        if self._config.use_whitened:
            self._posterior_log_density = self.whitened_posterior_log_density
        else:
            self._posterior_log_density = self.unwhitened_posterior_log_density

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        expected_members_types = [("use_whitened", bool)]
        self._check_members_types(config, expected_members_types)
        num_dim = sample_shape[0]
        assert_trees_all_equal(sample_shape, (num_dim,))
        num_grid_per_dim = int(np.sqrt(num_dim))
        if num_grid_per_dim * num_grid_per_dim != num_dim:
            msg = (
                "num_dim needs to be a square number for LogGaussianCoxPines "
                "density."
            )
            raise ValueError(msg)

        if not config.file_path:
            msg = "Please specify a path in config for the Finnish pines data csv."
            raise ValueError(msg)

    def get_pines_points(self, file_path):
        """Get the pines data points."""
        with open(file_path, "rt") as input_file:
            b = np.genfromtxt(input_file, delimiter=",", skip_header=1, usecols=(1, 2))
        return b

    def whitened_posterior_log_density(self, white: Array) -> Array:
        quadratic_term = -0.5 * jnp.sum(white**2)
        prior_log_density = self._white_gaussian_log_normalizer + quadratic_term
        latent_function = cp_utils.get_latents_from_white(
            white, self._mu_zero, self._cholesky_gram
        )
        log_likelihood = cp_utils.poisson_process_log_likelihood(
            latent_function, self._poisson_a, self._flat_bin_counts
        )
        return prior_log_density + log_likelihood

    def unwhitened_posterior_log_density(self, latents: Array) -> Array:
        white = cp_utils.get_white_from_latents(
            latents, self._mu_zero, self._cholesky_gram
        )
        prior_log_density = (
            -0.5 * jnp.sum(white * white) + self._unwhitened_gaussian_log_normalizer
        )
        log_likelihood = cp_utils.poisson_process_log_likelihood(
            latents, self._poisson_a, self._flat_bin_counts
        )
        return prior_log_density + log_likelihood

    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        output = jax.vmap(self._posterior_log_density)(x)
        density_state += self._is_target * x.shape[0]
        return output, density_state


class ChallengingTwoDimensionalMixture(LogDensity):
    """A challenging mixture of Gaussians in two dimensions.

    num_dim should be 2. config is unused in this case.
    """

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        del config
        assert_trees_all_equal(sample_shape, (2,))

    def raw_log_density(self, x: Array) -> Array:
        """A raw log density that we will then symmetrize."""
        mean_a = jnp.array([3.0, 0.0])
        mean_b = jnp.array([-2.5, 0.0])
        mean_c = jnp.array([2.0, 3.0])
        means = jnp.stack((mean_a, mean_b, mean_c), axis=0)
        cov_a = jnp.array([[0.7, 0.0], [0.0, 0.05]])
        cov_b = jnp.array([[0.7, 0.0], [0.0, 0.05]])
        cov_c = jnp.array([[1.0, 0.95], [0.95, 1.0]])
        covs = jnp.stack((cov_a, cov_b, cov_c), axis=0)
        log_weights = jnp.log(jnp.array([1.0 / 3, 1.0 / 3.0, 1.0 / 3.0]))
        l = jnp.linalg.cholesky(covs)
        y = slinalg.solve_triangular(l, x[None, :] - means, lower=True, trans=0)
        mahalanobis_term = -1 / 2 * jnp.einsum("...i,...i->...", y, y)
        n = means.shape[-1]
        normalizing_term = -n / 2 * np.log(2 * np.pi) - jnp.log(
            l.diagonal(axis1=-2, axis2=-1)
        ).sum(axis=1)
        individual_log_pdfs = mahalanobis_term + normalizing_term
        mixture_weighted_pdfs = individual_log_pdfs + log_weights
        return logsumexp(mixture_weighted_pdfs)

    def make_2d_invariant(self, log_density, x: Array) -> Array:
        density_a = log_density(x)
        density_b = log_density(np.flip(x))
        return jnp.logaddexp(density_a, density_b) - jnp.log(2)

    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        density_func = lambda x: self.make_2d_invariant(self.raw_log_density, x)
        output = jax.vmap(density_func)(x)
        density_state += self._is_target * x.shape[0]
        return output, density_state


class AutoEncoderLikelihood(LogDensity):
    """Generative decoder log p(x,z| theta) as a function of latents z.

    This evaluates log p(x,z| theta) = log p(x, z| theta ) + log p(z) for a VAE.
    Here x is an binarized MNIST Image, z are real valued latents, theta denotes
    the generator neural network parameters.

    Since x is fixed and z is a random variable this is the log of an unnormalized
    z density p(x, z | theta)
    The normalizing constant is a marginal p(x | theta) = int p(x, z | theta) dz.
    The normalized target density is the posterior over latents p(z|x, theta).

    The likelihood uses a pretrained generator neural network.
    It is contained in a pickle file specifed by config.params_filesname

    A script producing such a pickle file can be found in train_vae.py

    The resulting pretrained network used in the AFT paper
    can be found at data/vae.pickle

    The binarized MNIST test set image used is specfied by config.image_index

    """

    def __init__(
        self, config: ConfigDict, sample_shape: SampleShape, is_target: bool = False
    ):
        super().__init__(config, sample_shape, is_target)
        self._num_dim = sample_shape[0]
        self._vae_params = self._get_vae_params(config.params_filename)
        test_batch_size = 1
        test_ds = vae_lib.load_dataset(tfds.Split.TEST, test_batch_size)
        for unused_index in range(self._config.image_index):
            unused_batch = next(test_ds)
        self._test_image = next(test_ds)["image"]
        assert self._test_image.shape[0] == 1  # Batch size needs to be 1.
        assert self._test_image.shape[1:] == vae_lib.MNIST_IMAGE_SHAPE
        self.entropy_eval = hk.transform(self.cross_entropy_eval_func)

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        assert_trees_all_equal(sample_shape, (30,))
        num_mnist_test = 10000
        in_range = config.image_index >= 0 and config.image_index < num_mnist_test
        if not in_range:
            msg = "VAE image_index must be greater than or equal to zero "
            msg += "and strictly less than " + str(num_mnist_test) + "."
            raise ValueError(msg)

    def _get_vae_params(self, ckpt_filename):
        with open(ckpt_filename, "rb") as f:
            vae_params = pickle.load(f)
        return vae_params

    def cross_entropy_eval_func(self, data: Array, latent: Array) -> Array:
        """Evaluate the binary cross entropy for given latent and data.

        Needs to be called within a Haiku transform.

        Args:
          data: Array of shape (1, image_shape)
          latent: Array of shape (num_latent_dim,)

        Returns:
          Array, value of binary cross entropy for single data point in question.
        """
        chex.assert_rank(latent, 1)
        chex.assert_rank(data, 4)  # Shape should be (1, 28, 28, 1) hence rank 4.
        vae = vae_lib.ConvVAE()
        # New axis here required for batch size = 1 for VAE compatibility.
        batch_latent = latent[None, :]
        logits = vae.decoder(batch_latent)
        chex.assert_equal_shape([logits, data])
        return vae_lib.binary_cross_entropy_from_logits(logits, data)

    def log_prior(self, latent: Array) -> Array:
        """Latent shape (num_dim,) -> standard multivariate log density."""
        chex.assert_rank(latent, 1)
        log_norm_gaussian = -0.5 * self._num_dim * jnp.log(2.0 * jnp.pi)
        data_term = -0.5 * jnp.sum(jnp.square(latent))
        return data_term + log_norm_gaussian

    def total_log_probability(self, latent: Array) -> Array:
        chex.assert_rank(latent, 1)
        log_prior = self.log_prior(latent)
        dummy_rng_key = 0
        # Data point log likelihood is negative of loss for batch size of 1.
        log_likelihood = -1.0 * self.entropy_eval.apply(
            self._vae_params, dummy_rng_key, self._test_image, latent
        )
        total_log_probability = log_prior + log_likelihood
        return total_log_probability

    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        output = jax.vmap(self.total_log_probability)(x)
        density_state += self._is_target * x.shape[0]
        return output, density_state


def phi_four_log_density(x: Array, mass_squared: Array, bare_coupling: Array) -> Array:
    """Evaluate the phi_four_log_density.

    Args:
      x: Array of size (L_x, L_y)- values on 2D lattice.
      mass_squared: Scalar representing bare mass squared.
      bare_coupling: Scare representing bare coupling.

    Returns:
      Scalar corresponding to log_density.
    """
    chex.assert_rank(x, 2)
    chex.assert_rank(mass_squared, 0)
    chex.assert_rank(bare_coupling, 0)
    mass_term = mass_squared * jnp.sum(jnp.square(x))
    quadratic_term = bare_coupling * jnp.sum(jnp.power(x, 4))
    roll_x_plus = jnp.roll(x, shift=1, axis=0)
    roll_x_minus = jnp.roll(x, shift=-1, axis=0)
    roll_y_plus = jnp.roll(x, shift=1, axis=1)
    roll_y_minus = jnp.roll(x, shift=-1, axis=1)
    # D'alembertian operator acting on field phi.
    dalembert_phi = 4.0 * x - roll_x_plus - roll_x_minus - roll_y_plus - roll_y_minus
    kinetic_term = jnp.sum(x * dalembert_phi)
    action_density = kinetic_term + mass_term + quadratic_term
    return -action_density


class PhiFourTheory(LogDensity):
    """Log density for phi four field theory in two dimensions."""

    def __init__(
        self, config: ConfigDict, sample_shape: SampleShape, is_target: bool = False
    ):
        super().__init__(config, sample_shape, is_target)
        self._num_grid_per_dim = int(np.sqrt(sample_shape[0]))

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        expected_members_types = [("mass_squared", float), ("bare_coupling", float)]
        num_dim = sample_shape[0]
        self._check_members_types(config, expected_members_types)
        num_grid_per_dim = int(np.sqrt(num_dim))
        if num_grid_per_dim * num_grid_per_dim != num_dim:
            msg = "num_dim needs to be a square number for PhiFourTheory " "density."
            raise ValueError(msg)

    def reshape_and_call(self, x: Array) -> Array:
        return phi_four_log_density(
            jnp.reshape(x, (self._num_grid_per_dim, self._num_grid_per_dim)),
            self._config.mass_squared,
            self._config.bare_coupling,
        )

    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        output = jax.vmap(self.reshape_and_call)(x)
        density_state += self._is_target * x.shape[0]
        return output, density_state


class ManyWell(LogDensity):
    """Many well log density.

    See:
      Midgeley, Stimper et al. Flow Annealed Importance Sampling Bootstrap. 2022.
      Wu et al. Stochastic Normalizing Flows. 2020.
    """

    def __init__(
        self, config: ConfigDict, sample_shape: SampleShape, is_target: bool = False
    ):
        super().__init__(config, sample_shape, is_target)
        self._num_dim = sample_shape[0]

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        num_dim = sample_shape[0]
        if num_dim % 2 != 0:
            msg = "sample_shape[0] needs to be even."
            raise ValueError(msg)

    def single_well_log_density(self, x) -> Array:
        chex.assert_shape(x, (2,))
        # Here we index from 0 instead of 1 which differs from the cited papers.
        x_zero_term = -1.0 * jnp.power(x[0], 4) + 6.0 * jnp.power(x[0], 2) + 0.5 * x[0]
        x_one_term = -0.5 * jnp.power(x[1], 2)
        return x_zero_term + x_one_term

    def many_well_log_density(self, x: Array) -> Array:
        chex.assert_rank(x, 2)
        per_group_log_densities = jax.vmap(self.single_well_log_density)(x)
        return jnp.sum(per_group_log_densities)

    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        chex.assert_rank(x, 2)
        (num_batch, num_dim) = x.shape
        reshaped_x = jnp.reshape(x, (num_batch, num_dim // 2, 2))
        output = jax.vmap(self.many_well_log_density)(reshaped_x)
        density_state += self._is_target * x.shape[0]
        return output, density_state


class BayesianLogisticRegression(LogDensity):
    """Evalute the unnormalised log posterior
    for a bayesian logistic regression model:
    theta \sim N(0, I)
    y | x, theta \sim Bernoulli(sigmoid(theta^T x))

    Implementation adapted from Denoising Diffusion Samplers
    https://arxiv.org/pdf/2302.13834.pdf
    """

    def __init__(
        self, config: ConfigDict, sample_shape: SampleShape, is_target: bool = False
    ):
        def model(y_obs):
            w = numpyro.sample(
                "weights", numpyro.distributions.Normal(np.zeros(dim), np.ones(dim))
            )
            logits = jnp.dot(x, w)
            with numpyro.plate("J", n_data):
                _ = numpyro.sample(
                    "y", numpyro.distributions.BernoulliLogits(logits), obs=y_obs
                )

        x, y_ = load_data(config.file_path)
        dim = x.shape[1]
        n_data = x.shape[0]
        model_args = (y_,)

        rng_key = jax.random.PRNGKey(1)
        model_param_info, potential_fn, _, _ = numpyro.infer.util.initialize_model(
            rng_key, model, model_args=model_args
        )
        params_flat, unflattener = ravel_pytree(model_param_info[0])

        self.log_prob_model = lambda z: -1.0 * potential_fn(unflattener(z))
        dim = params_flat.shape[0]

        super().__init__(config, sample_shape, is_target)

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        if "ion" in config.file_path:
            assert_trees_all_equal(sample_shape, (35,))
        if "sonar" in config.file_path:
            assert_trees_all_equal(sample_shape, (61,))

    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        output = jax.vmap(self.log_prob_model, in_axes=0)(x)
        density_state += self._is_target * x.shape[0]
        return output, density_state


class BrownianMissingMiddleScales(LogDensity):
    def __init__(
        self, config: ConfigDict, sample_shape: SampleShape, is_target: bool = False
    ):
        self.observed_locs = np.array(
            [
                0.21592641,
                0.118771404,
                -0.07945447,
                0.037677474,
                -0.27885845,
                -0.1484156,
                -0.3250906,
                -0.22957903,
                -0.44110894,
                -0.09830782,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.8786016,
                -0.83736074,
                -0.7384849,
                -0.8939254,
                -0.7774566,
                -0.70238715,
                -0.87771565,
                -0.51853573,
                -0.6948214,
                -0.6202789,
            ]
        ).astype(dtype=np.float32)

        super().__init__(config, sample_shape, is_target)

    def _check_constructor_inputs(self, config: ConfigDict, sample_shape: SampleShape):
        assert_trees_all_equal(sample_shape, (32,))

    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        def unbatched(x_):
            log_jacobian_term = -jnp.log(1 + jnp.exp(-x_[0])) - jnp.log(
                1 + jnp.exp(-x_[1])
            )
            x_ = x_.at[0].set(jnp.log(1 + jnp.exp(x_[0])))
            x_ = x_.at[1].set(jnp.log(1 + jnp.exp(x_[1])))
            inn_noise_prior = jax.scipy.stats.norm.logpdf(
                jnp.log(x_[0]), loc=0.0, scale=2
            ) - jnp.log(x_[0])
            obs_noise_prior = jax.scipy.stats.norm.logpdf(
                jnp.log(x_[1]), loc=0.0, scale=2
            ) - jnp.log(x_[1])
            hidden_loc_0_prior = jax.scipy.stats.norm.logpdf(
                x_[2], loc=0.0, scale=x_[0]
            )
            hidden_loc_priors = hidden_loc_0_prior
            for i in range(29):
                hidden_loc_priors += jax.scipy.stats.norm.logpdf(
                    x_[i + 3], loc=x_[i + 2], scale=x_[0]
                )
            log_prior = inn_noise_prior + obs_noise_prior + hidden_loc_priors

            inds_not_nan = np.argwhere(~np.isnan(self.observed_locs)).flatten()
            log_lik = jax.vmap(
                lambda x, y: jax.scipy.stats.norm.logpdf(y, loc=x, scale=x_[1])
            )(x_[inds_not_nan + 2], self.observed_locs[inds_not_nan])

            log_posterior = log_prior + jnp.sum(log_lik)

            return log_posterior + log_jacobian_term

        if len(x.shape) == 1:
            density_state += self._is_target
            return unbatched(x), density_state
        else:
            density_state += self._is_target * x.shape[0]
            return jax.vmap(unbatched)(x), density_state
