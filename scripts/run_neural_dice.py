# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os
import d4rl ## needed for maze2d envs
import numpy as np
import tensorflow as tf
# import wandb

from tf_agents import specs
from tf_agents.environments import suite_mujoco, tf_py_environment
from tf_agents.policies.actor_policy import ActorPolicy

from dice_rl.estimators.neural_dice import NeuralDice
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
from dice_rl.data.dataset import Dataset

from dice_rl.networks.torch_actor_network import TorchActorNetwork


FLAGS = flags.FLAGS

flags.DEFINE_string('buffer_filepath', None, 'Directory to load dataset from.')
flags.DEFINE_string('policy_filepath', None, 'Directory to load policy from')
flags.DEFINE_string('save_dir', None,
                    'Directory to save the model and estimation results.')
flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_bool('tabular_obs', False, 'Whether to use tabular observations.')
flags.DEFINE_integer('num_trajectory', 1000,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 40,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_float('nu_learning_rate', 0.0001, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.0001, 'Learning rate for zeta.')
flags.DEFINE_float('nu_regularizer', 0.0, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.0, 'Ortho regularization on zeta.')
flags.DEFINE_integer('num_steps', 500000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 2048, 'Batch size.')

flags.DEFINE_float('f_exponent', 2, 'Exponent for f function.')
flags.DEFINE_bool('primal_form', False,
                  'Whether to use primal form of loss for nu.')

flags.DEFINE_float('primal_regularizer', 0.,
                   'LP regularizer of primal variables.')
flags.DEFINE_float('dual_regularizer', 1., 'LP regularizer of dual variables.')
flags.DEFINE_bool('zero_reward', False,
                  'Whether to ignore reward in optimization.')
flags.DEFINE_float('norm_regularizer', 1.,
                   'Weight of normalization constraint.')
flags.DEFINE_bool('zeta_pos', True, 'Whether to enforce positivity constraint.')

flags.DEFINE_float('scale_reward', 1., 'Reward scaling factor.')
flags.DEFINE_float('shift_reward', 0., 'Reward shift factor.')
flags.DEFINE_string(
    'transform_reward', None, 'Non-linear reward transformation'
    'One of [exp, cuberoot, None]')

flags.DEFINE_string("algo_name", None, "Algorithm name.", required=True)
# flags.DEFINE_string("wandb_project_name", "ope_baselines", "Wandb project name.", required=False)
# flags.DEFINE_string("unique_id", "storm", "Unique identifier", required=False)


def get_target_policy_from_torch(load_dir, env_name):
  """
  Load target policy.
  """
  env = suite_mujoco.load(env_name)
  env = tf_py_environment.TFPyEnvironment(env)

  ## dummy action experiment
  if env_name.startswith("Pendulum"):
    action_spec = specs.tensor_spec.from_spec(
      specs.BoundedArraySpec(
        shape   = (2,),
        dtype   = np.float32,
        minimum = np.array([-2, -0], dtype=np.float32),
        maximum = np.array([2, 0], dtype=np.float32),
        name    = "action"
      )
    )
  else:
    action_spec = env.action_spec()

  actor_net = TorchActorNetwork(
    input_tensor_spec   = env.observation_spec(),
    output_tensor_spec  = action_spec,
    fc_layer_params     = (256, 256)
  )

  ## need to build actor_net first by constructing ActorPolicy
  policy = ActorPolicy(
    time_step_spec  = env.time_step_spec(),
    action_spec     = action_spec,
    actor_network   = actor_net,
    training        = False
  )

  ## policy keeps reference to actor_net, so it automatically updates
  actor_net.load_torch_weights(load_dir, is_pendulum=env_name.startswith("Pendulum"))
  return policy


def main(argv):
  ## GPU is problematic for current installation of TF on storm servers
  ## but somehow converting dataset can use GPU...
  os.environ["CUDA_VISIBLE_DEVICES"] = ""

  buffer_filepath = FLAGS.buffer_filepath
  policy_filepath = FLAGS.policy_filepath
  save_dir = FLAGS.save_dir
  env_name = FLAGS.env_name
  algo_name = FLAGS.algo_name
  seed = FLAGS.seed
  gamma = FLAGS.gamma
  nu_learning_rate = FLAGS.nu_learning_rate
  zeta_learning_rate = FLAGS.zeta_learning_rate
  nu_regularizer = FLAGS.nu_regularizer
  zeta_regularizer = FLAGS.zeta_regularizer
  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size

  f_exponent = FLAGS.f_exponent
  primal_form = FLAGS.primal_form

  primal_regularizer = FLAGS.primal_regularizer
  dual_regularizer = FLAGS.dual_regularizer
  zero_reward = FLAGS.zero_reward
  norm_regularizer = FLAGS.norm_regularizer
  zeta_pos = FLAGS.zeta_pos

  scale_reward = FLAGS.scale_reward
  shift_reward = FLAGS.shift_reward
  transform_reward = FLAGS.transform_reward

  # wandb_project_name = FLAGS.wandb_project_name
  # unique_id = FLAGS.unique_id

  ## seeding
  tf.random.set_seed(seed)

  ## wandb config
  # os.environ["WANDB_API_KEY"] = "347cf7dc2e58b4eed9a6211c618daff1ba02cdfa"

  # wandb_config = {
  #   "seed": seed,
  #   "env_name": env_name,
  #   "gamma": gamma,
  #   "nu_learning_rate": nu_learning_rate,
  #   "zeta_learning_rate": zeta_learning_rate,
  #   "nu_regularizer": nu_regularizer,
  #   "zeta_regularizer": zeta_regularizer,
  #   "num_steps": num_steps,
  #   "batch_size": batch_size,
  #   "f_exponent": f_exponent,
  #   "primal_form": primal_form,
  #   "primal_regularizer": primal_regularizer,
  #   "dual_regularizer": dual_regularizer,
  #   "zero_reward": zero_reward,
  #   "norm_regularizer": norm_regularizer,
  #   "zeta_pos": zeta_pos,
  #   "scale_reward": scale_reward,
  #   "shift_reward": shift_reward,
  #   "transform_reward": transform_reward,
  #   "unique_id": unique_id
  # }
  # run = wandb.init(
  #   mode    = "online",
  #   entity  = "kernel_metric",
  #   project = wandb_project_name,
  #   group   = algo_name,
  #   dir     = save_dir,
  #   config  = wandb_config
  # )

  def reward_fn(env_step):
    reward = env_step.reward * scale_reward + shift_reward
    if transform_reward is None:
      return reward
    if transform_reward == 'exp':
      reward = tf.math.exp(reward)
    elif transform_reward == 'cuberoot':
      reward = tf.sign(reward) * tf.math.pow(tf.abs(reward), 1.0 / 3.0)
    else:
      raise ValueError('Reward {} not implemented.'.format(transform_reward))
    return reward

  if save_dir is not None:
    save_dir = os.path.join(save_dir, env_name, algo_name, "seed-%d_gamma-%.3f" % (seed, gamma))
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
    summary_writer.set_as_default()
  else:
    tf.summary.create_noop_writer()

  print('Loading dataset from', buffer_filepath)
  dataset = Dataset.load(buffer_filepath)
  all_steps = dataset.get_all_steps()
  max_reward = tf.reduce_max(all_steps.reward)
  min_reward = tf.reduce_min(all_steps.reward)
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)
  print('min reward', min_reward, 'max reward', max_reward)
  print('behavior per-step',
        estimator_lib.get_fullbatch_average(dataset, gamma=gamma))

  activation_fn = tf.nn.relu
  kernel_initializer = tf.keras.initializers.GlorotUniform()
  hidden_dims = (64, 64)
  input_spec = (dataset.spec.observation, dataset.spec.action)
  nu_network = ValueNetwork(
      input_spec,
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      kernel_initializer=kernel_initializer,
      last_kernel_initializer=kernel_initializer)
  output_activation_fn = tf.math.square if zeta_pos else tf.identity
  zeta_network = ValueNetwork(
      input_spec,
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      output_activation_fn=output_activation_fn,
      kernel_initializer=kernel_initializer,
      last_kernel_initializer=kernel_initializer)

  nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)
  zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate, clipvalue=1.0)
  lam_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)

  estimator = NeuralDice(
      dataset_spec        = dataset.spec,
      nu_network          = nu_network,
      zeta_network        = zeta_network,
      nu_optimizer        = nu_optimizer,
      zeta_optimizer      = zeta_optimizer,
      lam_optimizer       = lam_optimizer,
      gamma               = gamma,
      zero_reward         = zero_reward,
      f_exponent          = f_exponent,
      primal_form         = primal_form,
      reward_fn           = reward_fn,
      primal_regularizer  = primal_regularizer,
      dual_regularizer    = dual_regularizer,
      norm_regularizer    = norm_regularizer,
      nu_regularizer      = nu_regularizer,
      zeta_regularizer    = zeta_regularizer
  )

  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)

  target_policy = get_target_policy_from_torch(policy_filepath, env_name)
  running_losses = []
  running_estimates = []
  for step in range(num_steps):
    transitions_batch = dataset.get_step(batch_size, num_steps=2)
    initial_steps_batch, _ = dataset.get_episode(
        batch_size, truncate_episode_at=1)
    initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
                                                initial_steps_batch)
    losses = estimator.train_step(initial_steps_batch, transitions_batch,
                                  target_policy)
    running_losses.append(losses)
    if step % 1000 == 0 or step == num_steps - 1:
      estimate = estimator.estimate_average_reward(dataset, target_policy, None)# run)
      running_estimates.append(estimate)
      running_losses = []
    global_step.assign_add(1)

  print('Done!')


if __name__ == '__main__':
  app.run(main)
