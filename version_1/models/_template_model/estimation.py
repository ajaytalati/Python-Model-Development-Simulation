"""_template_model/estimation.py — EstimationModel stub.

Reference:  how_to_add_a_new_model/02_estimation.md

This file is NOT required for simulation-only usage.  Fill it in only
when you need Bayesian parameter inference (particle-filter, SMC2, EKF).

The skeleton below shows the expected structure:

  * PARAM_PRIOR_CONFIG — OrderedDict of estimated parameter priors.
  * INIT_STATE_PRIOR_CONFIG — OrderedDict of estimated initial-state priors.
  * FROZEN_PARAMS — dict of parameters used by the simulator but NOT
    estimated (e.g. fixed constants, noise temperatures the sampler can't
    identify).
  * The JAX callbacks (propagate_fn, obs_log_weight_fn, align_obs_fn,
    shard_init_fn) — see 02_estimation.md §5 for signatures and examples.

See models/sleep_wake_20p/estimation.py for a complete working example.
"""

from collections import OrderedDict
import math


# =========================================================================
# PRIORS
# =========================================================================

# OrderedDict keys define the parameter-vector ordering used by the sampler.
# Distribution shorthand: ('lognormal' | 'normal' | 'vonmises' | 'beta',
#                          (args...))
#
# Example entries for the OU template:
PARAM_PRIOR_CONFIG = OrderedDict([
    ('k',         ('lognormal', (math.log(1.0),  0.3))),
    ('mu',        ('normal',    (0.0, 0.5))),
    ('sigma_x',   ('lognormal', (math.log(0.5),  0.3))),
    ('sigma_obs', ('lognormal', (math.log(0.2),  0.3))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('x_0', ('normal', (2.0, 1.0))),
])

FROZEN_PARAMS = {
    # Non-estimated scalars consumed by the simulator.
    # (dt_hours and t_total_hours typically go here.)
}


# =========================================================================
# JAX CALLBACKS (to be implemented)
# =========================================================================
#
# The EstimationModel object expects the following callables.  See
# 02_estimation.md §5 for signatures.  The skeleton is commented out
# because it requires JAX; implement once you are ready to estimate.
#
# def propagate_fn(y, theta, key, dt):
#     ...
# def obs_log_weight_fn(y, y_obs, theta):
#     ...
# def align_obs_fn(channel_outputs, t_grid, params):
#     ...
# def shard_init_fn(theta, key, n_particles):
#     ...
#
# When they exist, build the model object:
#
# from estimation_model import EstimationModel
# TEMPLATE_ESTIMATION = EstimationModel(
#     name="_template_",
#     version="0.1",
#     n_states=1, n_stochastic=1, stochastic_indices=(0,),
#     state_bounds=((-10.0, 10.0),),
#     param_prior_config=PARAM_PRIOR_CONFIG,
#     init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
#     frozen_params=FROZEN_PARAMS,
#     propagate_fn=propagate_fn,
#     obs_log_weight_fn=obs_log_weight_fn,
#     align_obs_fn=align_obs_fn,
#     shard_init_fn=shard_init_fn,
# )
