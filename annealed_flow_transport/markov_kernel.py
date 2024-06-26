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

"""Code for Markov transition kernels."""

from typing import Tuple

import annealed_flow_transport.aft_types as tp
import chex
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

mcmc = tfp.mcmc
ConfigDict = tp.ConfigDict
Array = tp.Array
LogDensityByStep = tp.LogDensityByStep
RandomKey = tp.RandomKey
MarkovKernelApply = tp.MarkovKernelApply
Samples = tp.Samples
assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes


class InterpolatedStepSize(object):
    """Interpolate MCMC step sizes."""

    def __init__(self, config: ConfigDict, total_num_time_steps: int):
        self._config = config
        self._total_num_time_steps = total_num_time_steps

    def __call__(self, time_step: int):
        final_step = self._total_num_time_steps - 1.0
        beta = time_step / final_step
        return jnp.interp(
            beta, jnp.array(self._config.step_times), jnp.array(self._config.step_sizes)
        )


def tree_add(tree_a, tree_b):
    assert_trees_all_equal_shapes(tree_a, tree_b)
    return jax.tree_map(lambda a, b: a + b, tree_a, tree_b)


def tree_scalar_mul(tree, scalar):
    res = jax.tree_map(lambda x: x * scalar, tree)
    assert_trees_all_equal_shapes(tree, res)
    return res


def random_walk_metropolis(
    samples_in: Array,
    proposal_scale: Array,
    log_density_by_step: LogDensityByStep,
    temp_step: int,
    num_mh_steps: int,
    key: RandomKey,
    density_state: int,
) -> Tuple[Array, Array, int]:
    """Corrected random walk Metropolis-Hastings algorithm.

    Args:
      samples_in: (num_batch, num_dim)
      proposal_scale: Scalar representing scale of isotropic normal proposal.
      log_density_by_step: Target log density.
      temp_step: Step of outer annealing algorithm.
      num_mh_steps: Number of Metropolis-Hastings steps.
      key: Jax Random Key.
    Returns:
      samples_out: (num_batch, num_dim)
      acceptance: Average acceptance rate of chains.
    """
    chex.assert_rank(proposal_scale, 0)
    num_batch = np.shape(jax.tree_util.tree_leaves(samples_in)[0])[0]

    def rwm_step(state_in, curr_key: RandomKey):
        previous_samples = state_in[0]
        density_state = state_in[1]
        normal_key, acceptance_key = jax.random.split(curr_key)
        standard_normal_tree = random_normal_like_tree(normal_key, previous_samples)
        normal_deltas = tree_scalar_mul(standard_normal_tree, proposal_scale)
        exponential_rvs = jax.random.exponential(key=acceptance_key, shape=(num_batch,))
        proposed_samples = tree_add(previous_samples, normal_deltas)
        assert_trees_all_equal_shapes(previous_samples, proposed_samples)
        log_density_proposed, density_state = log_density_by_step(
            temp_step, proposed_samples, density_state
        )
        log_density_previous, density_state = log_density_by_step(
            temp_step, previous_samples, density_state
        )
        delta_log_prob = log_density_proposed - log_density_previous
        chex.assert_shape(delta_log_prob, (num_batch,))
        is_accepted = jnp.greater(delta_log_prob, -1.0 * exponential_rvs)
        chex.assert_shape(is_accepted, (num_batch,))
        step_acceptance_rate = jnp.mean(is_accepted * 1.0)
        samples_next = jnp.where(
            is_accepted[:, None], proposed_samples, previous_samples
        )
        return (
            (samples_next, density_state),
            step_acceptance_rate,
        )

    keys = jax.random.split(key, num_mh_steps)
    state_out, acceptance_rates = jax.lax.scan(
        rwm_step, (samples_in, density_state), keys
    )
    samples_out = state_out[0]
    density_state = state_out[1]
    acceptance_rate = jnp.mean(acceptance_rates)
    chex.assert_equal_shape((samples_out, samples_in))
    chex.assert_rank(acceptance_rate, 0)
    return samples_out, acceptance_rate, density_state


def momentum_step(
    samples_in: Array,
    momentum_in: Array,
    step_coefficient: Array,
    epsilon: Array,
    grad_log_density,
    density_state: int,
) -> tp.Tuple[Array, int]:
    """A momentum update with variable momentum step_coefficient.

    Args:
      samples_in: (num_batch, num_dim)
      momentum_in: (num_batch, num_dim)
      step_coefficient: A Scalar which is typically either 0.5 (half step) or 1.0
      epsilon: A Scalar representing the constant step size.
      grad_log_density: (num_batch, num_dim) -> (num_batch, num_dim)
    Returns:
      momentum_out (num_batch, num_dim)
    """
    chex.assert_rank((step_coefficient, epsilon), (0, 0))
    assert_trees_all_equal_shapes(samples_in, momentum_in)
    gradient_val, density_state = grad_log_density(samples_in, density_state)
    momentum_out = tree_add(
        momentum_in, tree_scalar_mul(gradient_val, step_coefficient * epsilon)
    )
    assert_trees_all_equal_shapes(momentum_in, momentum_out)
    return momentum_out, density_state


def leapfrog_step(
    samples_in: Array,
    momentum_in: Array,
    step_coefficient: Array,
    epsilon: Array,
    grad_log_density,
    density_state: int,
) -> Tuple[Array, Array]:
    """A step of the Leapfrog iteration with variable momentum step_coefficient.

    Args:
      samples_in: (num_batch, num_dim)
      momentum_in: (num_batch, num_dim)
      step_coefficient: A Scalar which is typically either 0.5 (half step) or 1.0
      epsilon: A Scalar representing the constant step size.
      grad_log_density: (num_batch, num_dim) -> (num_batch, num_dim)
    Returns:
      samples_out: (num_batch, num_dim)
      momentum_out (num_batch, num_dim)
    """
    chex.assert_rank((step_coefficient, epsilon), (0, 0))
    assert_trees_all_equal_shapes(samples_in, momentum_in)
    samples_out = tree_add(samples_in, tree_scalar_mul(momentum_in, epsilon))
    momentum_out, density_state = momentum_step(
        samples_out,
        momentum_in,
        step_coefficient,
        epsilon,
        grad_log_density,
        density_state,
    )
    assert_trees_all_equal_shapes(samples_in, samples_out)
    return samples_out, momentum_out, density_state


def random_normal_like_tree(key, tree):
    tree_struct = jax.tree_util.tree_structure(tree)
    split_keys = jax.random.split(key, tree_struct.num_leaves)
    tree_keys = jax.tree_util.tree_unflatten(tree_struct, split_keys)
    tree_normals = jax.tree_util.tree_map(
        lambda x, y: jax.random.normal(key=y, shape=x.shape), tree, tree_keys
    )
    return tree_normals


def hmc_step(
    samples_in: Array,
    key: RandomKey,
    epsilon: Array,
    log_density,
    grad_log_density,
    num_leapfrog_iters: int,
    density_state: int,
) -> Tuple[Array, Array]:
    """A single step of Hamiltonian Monte Carlo.

    Args:
      samples_in: (num_batch, num_dim)
      key: A Jax random key.
      epsilon: A Scalar representing the constant step size.
      log_density: (num_batch, num_dim) -> (num_batch,)
      grad_log_density: (num_batch, num_dim) -> (num_batch, num_dim)
      num_leapfrog_iters: Number of leapfrog iterations.
    Returns:
      samples_out: (num_batch, num_dim)
    """
    chex.assert_rank(epsilon, 0)
    samples_state = samples_in
    momentum_key, acceptance_key = jax.random.split(key)
    initial_momentum = random_normal_like_tree(momentum_key, samples_in)
    # A half momentum step.
    momentum_state, density_state = momentum_step(
        samples_state,
        initial_momentum,
        step_coefficient=0.5,
        epsilon=epsilon,
        grad_log_density=grad_log_density,
        density_state=density_state,
    )

    def scan_step(passed_state, unused_input):
        pos, mom, density_state = passed_state
        new_pos, new_mom, density_state = leapfrog_step(
            pos,
            mom,
            step_coefficient=1.0,
            epsilon=epsilon,
            grad_log_density=grad_log_density,
            density_state=density_state,
        )
        return (new_pos, new_mom, density_state), None

    state_in = (samples_state, momentum_state, density_state)
    scan_length = num_leapfrog_iters - 1
    # (num_leapfrog_iters - 1) whole position and momentum steps.
    new_state, _ = jax.lax.scan(
        scan_step, state_in, [None] * scan_length, length=scan_length
    )
    samples_state, momentum_state, density_state = new_state

    # A whole position step and half momentum step.
    samples_state, momentum_state, density_state = leapfrog_step(
        samples_state,
        momentum_state,
        step_coefficient=0.5,
        epsilon=epsilon,
        grad_log_density=grad_log_density,
        density_state=density_state,
    )

    # We don't negate the momentum here because it has no effect.
    # This would be required if momentum was used other than for just the energy.

    # Decide if we accept the proposed update using Metropolis correction.
    def get_combined_log_densities(pos, mom, density_state):
        pos_log_densities, density_state = log_density(pos, density_state)

        def leaf_log_density(x):
            summation_axes = tuple(range(1, len(np.shape(x))))
            return -0.5 * jnp.sum(jnp.square(x), axis=summation_axes)

        per_leaf_mom_log_densities = jax.tree_util.tree_map(leaf_log_density, mom)
        mom_log_densities = jax.tree_util.tree_reduce(
            jnp.add, per_leaf_mom_log_densities
        )
        chex.assert_equal_shape((pos_log_densities, mom_log_densities))
        return pos_log_densities + mom_log_densities, density_state

    current_log_densities, density_state = get_combined_log_densities(
        samples_in, initial_momentum, density_state
    )
    proposed_log_densities, density_state = get_combined_log_densities(
        samples_state, momentum_state, density_state
    )
    num_batch = np.shape(current_log_densities)[0]
    exponential_rvs = jax.random.exponential(key=acceptance_key, shape=(num_batch,))

    delta_log_prob = proposed_log_densities - current_log_densities
    chex.assert_shape(delta_log_prob, (num_batch,))
    is_accepted = jnp.greater(delta_log_prob, -1.0 * exponential_rvs)
    chex.assert_shape(is_accepted, (num_batch,))
    step_acceptance_rate = jnp.mean(is_accepted * 1.0)

    def acceptance(a, b):
        broadcast_axes = tuple(range(1, len(a.shape)))
        broadcast_is_accepted = jnp.expand_dims(is_accepted, axis=broadcast_axes)
        return jnp.where(broadcast_is_accepted, a, b)

    samples_next = jax.tree_util.tree_map(acceptance, samples_state, samples_in)
    return (samples_next, density_state), step_acceptance_rate


def hmc(
    samples_in: Array,
    key: RandomKey,
    epsilon: Array,
    log_density,
    grad_log_density,
    num_leapfrog_iters: int,
    num_hmc_iters: int,
    density_state: int,
) -> Tuple[Array, Array, int]:
    """Hamiltonian Monte Carlo as described in Neal 2011.

    Args:
      samples_in: (num_batch, num_dim)
      key: A Jax random key.
      epsilon: A Scalar representing the constant step size.
      log_density: (num_batch, num_dim) -> (num_batch,)
      grad_log_density: (num_batch, num_dim) -> (num_batch, num_dim)
      num_leapfrog_iters: Number of leapfrog iterations.
      num_hmc_iters: Number of steps of Hamiltonian Monte Carlo.
    Returns:
      samples_out: (num_batch, num_dim)
    """
    step_keys = jax.random.split(key, num_hmc_iters)

    def short_hmc_step(state_in, loc_key):
        loc_samples, density_state = state_in
        return hmc_step(
            loc_samples,
            loc_key,
            epsilon=epsilon,
            log_density=log_density,
            grad_log_density=grad_log_density,
            num_leapfrog_iters=num_leapfrog_iters,
            density_state=density_state,
        )

    (samples_final, density_state), acceptance_rates = jax.lax.scan(
        short_hmc_step, (samples_in, density_state), step_keys
    )

    return samples_final, np.mean(acceptance_rates), density_state


def hmc_wrapped(
    samples_in: Samples,
    key: RandomKey,
    epsilon: Array,
    log_density_by_step: LogDensityByStep,
    temp_step: int,
    num_leapfrog_iters: int,
    num_hmc_iters: int,
    density_state: int,
) -> Tuple[Array, Array]:
    """A wrapper for HMC that deals with all the interfacing with the codebase.

    Args:
      samples_in: Samples.
      key: A Jax random key.
      epsilon: Scalar step size.
      log_density_by_step: Density at a given temperature.
      temp_step: Specifies the current temperature.
      num_leapfrog_iters: Number of leapfrog iterations.
      num_hmc_iters: Number of Hamiltonian Monte Carlo iterations.
    Returns:
      tfp_samples_out: (0, num_batch, num_dim)
    """
    log_density = lambda x, density_state: log_density_by_step(
        temp_step, x, density_state
    )

    def grad_log_density(x, density_state):
        _, vjpfun, density_state = jax.vjp(
            lambda x: log_density(x=x, density_state=density_state), x, has_aux=True
        )
        grad = vjpfun(jnp.ones_like(x)[..., 0])[0]
        return grad, density_state

    # def unbatched_log_density(unbatched_tree_in):
    #     # Takes an unbatched tree and returns a single scalar value.
    #     batch_one_tree = jax.tree_util.tree_map(lambda x: x[None], unbatched_tree_in)
    #     return log_density(batch_one_tree)[0]

    # grad_log_density = jax.vmap(jax.grad(unbatched_log_density))

    samples_out, acceptance, density_state = hmc(
        samples_in,
        key=key,
        epsilon=epsilon,
        log_density=log_density,
        grad_log_density=grad_log_density,
        num_leapfrog_iters=num_leapfrog_iters,
        num_hmc_iters=num_hmc_iters,
        density_state=density_state,
    )
    return samples_out, acceptance, density_state


def nuts_wrapped(
    samples_in: Samples,
    key: RandomKey,
    epsilon: Array,
    log_density_by_step: LogDensityByStep,
    temp_step: int,
    num_nuts_iters: int,
) -> Tuple[Array, Array]:
    log_density = lambda x: log_density_by_step(temp_step, x, 0)[0]
    kernel = mcmc.NoUTurnSampler(log_density, epsilon)
    kernel_results_in = kernel.bootstrap_results(samples_in)
    for _ in range(num_nuts_iters):
        samples_out, kernel_results_out = kernel.one_step(
            samples_in, kernel_results_in, seed=key
        )
        samples_in, kernel_results_in = samples_out, kernel_results_out
    acceptance = jnp.exp(jnp.mean(kernel_results_out.log_accept_ratio))
    return samples_out, acceptance


class MarkovTransitionKernel(object):
    """Wraps TFP slice sampling and NUTS allowing configuration/composition."""

    def __init__(
        self,
        config: ConfigDict,
        density_by_step: LogDensityByStep,
        total_time_steps: int,
    ):
        self._config = config
        self._density_by_step = density_by_step
        if hasattr(config, "hmc_step_config"):
            self._hmc_step_size = InterpolatedStepSize(
                config.hmc_step_config, total_time_steps
            )
        if hasattr(config, "rwm_step_config"):
            self._rwm_step_size = InterpolatedStepSize(
                config.rwm_step_config, total_time_steps
            )
        if hasattr(config, "nuts_step_config"):
            self._nuts_step_size = InterpolatedStepSize(
                config.nuts_step_config, total_time_steps
            )

    def __call__(
        self, step: int, key: RandomKey, samples: Samples, density_state: int
    ) -> tp.Tuple[Array, tp.Tuple[float, float, float], int]:
        """A single step of slice sampling followed by NUTS.

        Args:
          step: The time step of the overall algorithm.
          key: A JAX random key.
          samples: The current samples.
        Returns:
          New samples.
        """

        if self._config.rwm_steps_per_iter != 0:
            subkey, key = jax.random.split(key)
            samples, rwm_acc, density_state = random_walk_metropolis(
                samples,
                self._rwm_step_size(step),
                self._density_by_step,
                step,
                self._config.rwm_steps_per_iter,
                subkey,
                density_state,
            )
        else:
            rwm_acc = 1.0
        if self._config.hmc_steps_per_iter != 0:
            samples, hmc_acc, density_state = hmc_wrapped(
                samples,
                key,
                self._hmc_step_size(step),
                self._density_by_step,
                step,
                self._config.hmc_num_leapfrog_steps,
                self._config.hmc_steps_per_iter,
                density_state,
            )
        else:
            hmc_acc = 1.0
        if self._config.nuts_steps_per_iter != 0:
            samples, nuts_acc = nuts_wrapped(
                samples,
                key,
                self._nuts_step_size(step),
                self._density_by_step,
                step,
                self._config.nuts_steps_per_iter,
            )  # not sure how to get stateful NUTS sampler with TF
        else:
            nuts_acc = 1.0

        acceptance_tuple = (hmc_acc, rwm_acc, nuts_acc)
        return samples, acceptance_tuple, density_state
