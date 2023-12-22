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

"""Training for all SMC and flow algorithms."""
from functools import partial
import os
import socket
from typing import Callable, Tuple

from absl import logging as log
import numpy as np
import tqdm
from annealed_flow_transport.loggers_pl import LoggerCollection
from annealed_flow_transport import aft
from annealed_flow_transport import craft
from annealed_flow_transport import densities
from annealed_flow_transport import flow_transport
from annealed_flow_transport import flows
from annealed_flow_transport import markov_kernel
from annealed_flow_transport import samplers
from annealed_flow_transport import serialize
from annealed_flow_transport import smc
from annealed_flow_transport import snf
from annealed_flow_transport import vi
import annealed_flow_transport.aft_types as tp
import chex
import haiku as hk
import jax
import optax
from hydra.utils import instantiate
from omegaconf import OmegaConf


# Type defs.
Array = tp.Array
OptState = tp.OptState
UpdateFn = tp.UpdateFn
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
LogDensityByStep = tp.LogDensityByStep
RandomKey = tp.RandomKey
AcceptanceTuple = tp.AcceptanceTuple
FreeEnergyAndGrad = tp.FreeEnergyAndGrad
FreeEnergyEval = tp.FreeEnergyEval
MarkovKernelApply = tp.MarkovKernelApply
SamplesTuple = tp.SamplesTuple
LogWeightsTuple = tp.LogWeightsTuple
VfesTuple = tp.VfesTuple
InitialSampler = tp.InitialSampler
LogDensityNoStep = tp.LogDensityNoStep
assert_equal_shape = chex.assert_equal_shape
AlgoResultsTuple = tp.AlgoResultsTuple


def get_optimizer(initial_learning_rate: float, boundaries_and_scales):
    """Get an optimizer possibly with learning rate schedule."""
    if boundaries_and_scales is None:
        return optax.adam(initial_learning_rate)
    else:
        schedule_fn = optax.piecewise_constant_schedule(
            initial_learning_rate, boundaries_and_scales
        )
        opt = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(schedule_fn),
            optax.scale(-1.0),
        )
        return opt


def value_or_none(value: str, config):
    if value in config:
        return config[value]
    else:
        return None


def boundaries_and_scales(method, config):
    if method + "_boundaries" in config:
        if method == "snf":
            snf_boundaries_and_scales = {}
            for key, value in zip(config.snf_boundaries, config.snf_scales):
                snf_boundaries_and_scales[key] = value
            return craft_boundaries_and_scales
        if method == "craft":
            craft_boundaries_and_scales = {}
            for key, value in zip(config.craft_boundaries, config.craft_scales):
                craft_boundaries_and_scales[key] = value
            return craft_boundaries_and_scales
    else:
        return None


def prepare_outer_loop(
    initial_sampler: InitialSampler,
    initial_log_density: Callable[[Array], Array],
    final_log_density: Callable[[Array], Array],
    flow_func: Callable[[Array], Tuple[Array, Array]],
    config,
    key,
    logger,
) -> AlgoResultsTuple:
    """Shared code outer loops then calls the outer loops themselves.

    Args:
      initial_sampler: Function for producing initial sample.
      initial_log_density: Function for evaluating initial log density.
      final_log_density: Function for evaluating final log density.
      flow_func: Flow function to pass to Haiku transform.
      config: experiment configuration.
    Returns:
      An AlgoResultsTuple containing the experiment results.

    """
    num_temps = config.num_steps + 1
    if is_annealing_algorithm(config.algo):
        density_by_step = flow_transport.GeometricAnnealingSchedule(
            initial_log_density, final_log_density, num_temps
        )
    if is_markov_algorithm(config.algo):
        markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
            config.mcmc_config, density_by_step, num_temps
        )

    flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
    key, subkey = jax.random.split(key)
    single_normal_sample = initial_sampler(
        subkey, config.batch_size, (config.num_dims,)
    )
    key, subkey = jax.random.split(key)
    flow_init_params = flow_forward_fn.init(subkey, single_normal_sample)

    if value_or_none("save_checkpoint", config):

        def save_checkpoint(params):
            return serialize.save_checkpoint(config.params_filename, params)

    else:
        save_checkpoint = None

    if config.algo == "vi":
        # Add a save_checkpoint function here to enable saving final state.
        opt = get_optimizer(config.optimization_config.vi_step_size, None)
        opt_init_state = opt.init(flow_init_params)
        results = vi.outer_loop_vi(
            initial_sampler=initial_sampler,
            opt_update=opt.update,
            opt_init_state=opt_init_state,
            flow_init_params=flow_init_params,
            flow_apply=flow_forward_fn.apply,
            key=key,
            initial_log_density=initial_log_density,
            final_log_density=final_log_density,
            config=config,
            save_checkpoint=save_checkpoint,
        )
    elif config.algo == "smc":
        eval_sampler = jax.jit(
            partial(
                smc.fast_outer_loop_smc,
                density_by_step=density_by_step,
                initial_sampler=initial_sampler,
                markov_kernel_by_step=markov_kernel_by_step,
                config=config,
            )
        )
        rng = jax.random.PRNGKey(config.seed)
        log_Z = np.zeros(config.num_smc_iters)
        for i in tqdm.tqdm(
            range(config.num_smc_iters), disable=(not config.progress_bars)
        ):
            rng, rng_ = jax.random.split(rng)
            results = eval_sampler(key=rng_)
            log_Z[i] = results.log_normalizer_estimate
        # Save normalising constant estimates (comment out when not doing a final evaluation run)
        if config.save_samples:
            np.savetxt(
                f"/data/ziz/not-backed-up/anphilli/diffusion_smc/benchmarking_results/{config.group}_{config.name}_smc_{config.num_steps}_{config.seed}.csv",
                log_Z,
            )
        if logger:
            logger.log_metrics(
                {"final_log_Z": np.mean(log_Z), "var_final_log_Z": np.var(log_Z)}, 0
            )
        results = smc.outer_loop_smc(
            density_by_step=density_by_step,
            initial_sampler=initial_sampler,
            markov_kernel_by_step=markov_kernel_by_step,
            key=key,
            config=config,
            logger=logger,
        )
    elif config.algo == "snf":
        opt = get_optimizer(
            config.optimization_config.snf_step_size,
            boundaries_and_scales("snf", config.optimization_config),
        )
        log_step_output = None
        results = snf.outer_loop_snf(
            flow_init_params=flow_init_params,
            flow_apply=flow_forward_fn.apply,
            density_by_step=density_by_step,
            markov_kernel_by_step=markov_kernel_by_step,
            initial_sampler=initial_sampler,
            key=key,
            opt=opt,
            config=config,
            log_step_output=log_step_output,
            save_checkpoint=save_checkpoint,
            logger=logger,
        )
    elif config.algo == "aft":
        opt = get_optimizer(config.optimization_config.aft_step_size, None)
        opt_init_state = opt.init(flow_init_params)
        # Add a log_step_output function here to enable non-trivial step logging.
        log_step_output = None
        results = aft.outer_loop_aft(
            opt_update=opt.update,
            opt_init_state=opt_init_state,
            flow_init_params=flow_init_params,
            flow_apply=flow_forward_fn.apply,
            density_by_step=density_by_step,
            markov_kernel_by_step=markov_kernel_by_step,
            initial_sampler=initial_sampler,
            key=key,
            config=config,
            log_step_output=log_step_output,
        )
    elif config.algo == "craft":
        opt = get_optimizer(
            config.optimization_config.craft_step_size,
            boundaries_and_scales("craft", config.optimization_config),
        )
        opt_init_state = opt.init(flow_init_params)
        print(flow_init_params)
        log_step_output = None
        results, transition_params = craft.outer_loop_craft(
            opt_update=opt.update,
            opt_init_state=opt_init_state,
            flow_init_params=flow_init_params,
            flow_apply=flow_forward_fn.apply,
            flow_inv_apply=None,
            density_by_step=density_by_step,
            markov_kernel_by_step=markov_kernel_by_step,
            initial_sampler=initial_sampler,
            key=key,
            config=config,
            log_step_output=log_step_output,
            save_checkpoint=save_checkpoint,
            logger=logger,
        )
        eval_sampler = jax.jit(
            partial(
                craft.craft_evaluation_loop,
                transition_params=transition_params,
                flow_apply=flow_forward_fn.apply,
                markov_kernel_apply=markov_kernel_by_step,
                initial_sampler=initial_sampler,
                log_density=density_by_step,
                config=config,
            )
        )
        rng = jax.random.PRNGKey(config.seed)
        log_Z = np.zeros(config.num_smc_iters)
        for i in tqdm.tqdm(
            range(config.num_smc_iters), disable=(not config.progress_bars)
        ):
            rng, rng_ = jax.random.split(rng)
            eval_results = eval_sampler(key=rng_)
            log_Z[i] = eval_results.log_normalizer_estimate
        # Save normalising constant estimates (comment out when not doing a final evaluation run)
        if config.save_samples:
            np.savetxt(
                f"/data/ziz/not-backed-up/anphilli/diffusion_smc/benchmarking_results/{config.group}_{config.name}_craft_{config.num_steps}_{config.seed}.csv",
                log_Z,
            )
        if logger:
            logger.log_metrics(
                {"final_log_Z": np.mean(log_Z), "var_final_log_Z": np.var(log_Z)}, 0
            )

    else:
        raise NotImplementedError
    return results


def is_flow_algorithm(algo_name):
    return algo_name in ("aft", "vi", "craft", "snf")


def is_markov_algorithm(algo_name):
    return algo_name in ("aft", "craft", "snf", "smc")


def is_annealing_algorithm(algo_name):
    return algo_name in ("aft", "craft", "snf", "smc")


def run_experiment(config) -> AlgoResultsTuple:
    """Run a SMC flow experiment.

    Args:
      config: experiment configuration.
    Returns:
      An AlgoResultsTuple containing the experiment results.
    """
    log.info("Starting up...")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"Run path: {run_path}")
    log.info(f"Hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, config.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    loggers = [instantiate(logger_cfg) for logger_cfg in config.logging.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    log_density_initial = getattr(densities, config.initial_config.density)(
        config.initial_config, (config.num_dims,)
    )
    log_density_final = getattr(densities, config.final_config.density)(
        config.final_config, (config.num_dims,)
    )
    initial_sampler = getattr(samplers, config.initial_sampler_config.initial_sampler)(
        config.initial_sampler_config
    )

    key = jax.random.PRNGKey(config.seed)

    if config.flow_config.type == "ComposedFlows":
        config.flow_config.flow_configs = [
            config.flow_config.base_flow_config
        ] * config.flow_config.num_stack

    def flow_func(x):
        if is_flow_algorithm(config.algo):
            flow = getattr(flows, config.flow_config.type)(config.flow_config)
            return flow(x)
        else:
            return None

    results = prepare_outer_loop(
        initial_sampler,
        log_density_initial,
        log_density_final,
        flow_func,
        config,
        key,
        logger,
    )

    logger.save()
    logger.finalize("success")
    return results
