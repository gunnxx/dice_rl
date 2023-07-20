"""
Convert our custom `ReplayBuffer` to `TFOffpolicyDataset`.
"""

import argparse
import d4rl
import gym
import numpy as np
import os.path
import tensorflow as tf
import torch
import tqdm

from tf_agents import specs
from tf_agents.environments import suite_mujoco, tf_py_environment
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils

from dice_rl.data.dataset import Dataset, EnvStep
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
from dice_rl.estimators.estimator import get_fullbatch_average

## must be here because of the pickle stuff
from replay_buffer_torch import ReplayBuffer
from TD3 import TD3


def add_episodes_to_dataset(episodes, valid_ids, write_dataset):
    """
    Function is copied from `scripts/create_dataset.py`.
    """
    num_episodes = 1 if tf.rank(valid_ids) == 1 else tf.shape(valid_ids)[0]
    for ep_id in tqdm.tqdm(range(num_episodes), "Adding episodes to ds"):
        if tf.rank(valid_ids) == 1:
            this_valid_ids = valid_ids
            this_episode = episodes
        else:
            this_valid_ids = valid_ids[ep_id, ...]
            this_episode = tf.nest.map_structure(
                lambda t: t[ep_id, ...], episodes
            )

        episode_length = tf.shape(this_valid_ids)[0]
        for step_id in range(episode_length):
            this_valid_id = this_valid_ids[step_id]
            this_step = tf.nest.map_structure(
                lambda t: t[step_id, ...], this_episode
            )
            if this_valid_id:
                write_dataset.add_step(this_step)


def get_args() -> argparse.Namespace:
    """
    Get argument.

    Return:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_buffer_path", type=str)
    parser.add_argument("--save_buffer_path", type=str)
    parser.add_argument("--load_policy_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    ## TODO
    load_path = args.load_buffer_path
    filename  = load_path.split("/")[-1]
    env_name  = load_path.split("/")[-2]
    save_path = os.path.join(args.save_buffer_path, filename[:-3])

    ## we add absorbing state in this implementation
    assert env_name.startswith("HalfCheetah") or env_name.startswith("Walker2d") or env_name.startswith("Hopper"), "Only for this for now."

    ## load our dataset
    read_dataset : ReplayBuffer = torch.load(load_path, map_location=torch.device("cpu"))

    ## load our behavior policy to add transition to absorbing state
    env = gym.make(env_name)
    policy = TD3(
        state_dim   = env.observation_space.shape[0],
        action_dim  = env.action_space.shape[0],
        max_action  = float(env.action_space.high[0]),
        device      = "cpu"
    )
    policy.load(args.load_policy_path)
    del env

    ## start_size serves as an upper-bound of the number of trajectories
    num_trajectory = read_dataset.start_size
    if env_name.startswith("maze2d"):
        max_trajectory_length = 150
    elif env_name.startswith("Pendulum"):
        max_trajectory_length = 200
    elif env_name.startswith("HalfCheetah") or env_name.startswith("Walker2d") or env_name.startswith("Hopper"):
        max_trajectory_length = 1000
    else:
        raise KeyError("env_name is not considered yet.")

    ## load environment to create spec
    ## NOTE: use `suite_mujoco` instead of `suite_gym` because it converts every `gym.spaces.Box` to `float32`
    env = suite_mujoco.load(env_name)
    env = tf_py_environment.TFPyEnvironment(env)

    ## create observation spec
    ## add absorbing states
    observation_spec = env.observation_spec()
    if env_name.startswith("HalfCheetah") or env_name.startswith("Walker2d") or env_name.startswith("Hopper"):
        observation_spec = specs.tensor_spec.from_spec(
            specs.BoundedArraySpec(
                shape   = (observation_spec.shape[0] + 1,),
                dtype   = np.float32,
                minimum = observation_spec.minimum,
                maximum = observation_spec.maximum,
                name    = observation_spec.name
            )
        )

    ## create action spec
    ## dummy action dimension for pendulum
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

    ## create the environment spec
    time_step_spec = time_step.time_step_spec(observation_spec)
    step_num_spec = specs.tensor_spec.from_spec(
        specs.BoundedArraySpec(
            shape   = [],
            dtype   = np.int64,
            minimum = 0,
            maximum = max_trajectory_length,
            name    = "step_num"
        )
    )

    write_dataset_spec = EnvStep(
        step_type   = time_step_spec.step_type,
        step_num    = step_num_spec,
        observation = observation_spec,
        action      = action_spec,
        reward      = time_step_spec.reward,
        discount    = time_step_spec.discount,
        policy_info = (), ## NOTE: just check the notebook
        env_info    = {},
        other_info  = {}
    )

    ## prepare the tf dataset
    write_dataset = TFOffpolicyDataset(
        spec     = write_dataset_spec,
        capacity = num_trajectory * (max_trajectory_length + 2)
    )

    ## "container" initialization
    episode = []            ## List[EnvStep]
    episodes = []           ## List[List[EnvStep]]
    episode_lengths = []    ## List[int]

    ## timestep initialization
    t = 0
    step_type = time_step.StepType.FIRST

    ## iterate through the dataset
    for i in tqdm.tqdm(range(read_dataset.size), "Converting steps"):
        ## get one step of data from the replay buffer
        state = read_dataset.state[i]
        action = read_dataset.action[i]
        next_state = read_dataset.next_state[i]
        reward = read_dataset.reward[i][0]
        not_done = read_dataset.not_done[i]

        if env_name.startswith("HalfCheetah") or env_name.startswith("Walker2d") or env_name.startswith("Hopper"):
            state = torch.cat((state, torch.tensor([0.])))

        ## convert to `EnvStep`
        ## using dicount of 1 because NeuralDICE use other parameters for the discount factor
        ## see `neural_dice.py` Line 193
        step = EnvStep(
            step_type   = step_type,
            step_num    = tf.cast(t, tf.int64),
            observation = tf.cast(state, tf.float32),
            action      = tf.cast(action, tf.float32),
            reward      = tf.cast(reward, tf.float32),
            discount    = tf.cast(1, tf.float32),
            policy_info = (), ## NOTE: just check the notebook
            env_info    = {},
            other_info  = {}
        )
        episode.append(step)

        ## update timestep
        t += 1
        step_type = time_step.StepType.MID

        ## terminal, add transition from terminal state to absorbing state
        ## and (self) transition from absorbing state to absorbing state
        if not not_done:
            ## get action at the terminal state from behavior policy
            ## NOTE: this is hard-coded for standard deviation of 0.3
            with torch.no_grad(): next_action = policy.actor.forward(next_state)
            next_action = next_action + torch.randn_like(next_action) * 0.3

            next_state = torch.cat((next_state, torch.tensor([0.])))
            step = EnvStep(
                step_type   = time_step.StepType.MID,
                step_num    = tf.cast(t, tf.int64),
                observation = tf.cast(next_state, tf.float32),
                action      = tf.cast(next_action, tf.float32),
                reward      = tf.cast(0., tf.float32),
                discount    = tf.cast(1, tf.float32),
                policy_info = (), ## NOTE: just check the notebook
                env_info    = {},
                other_info  = {}
            )
            episode.append(step)

            absorbing_state = torch.zeros_like(next_state)
            absorbing_state[-1] = 1.
            step = EnvStep(
                step_type   = time_step.StepType.MID,
                step_num    = tf.cast(t, tf.int64),
                observation = tf.cast(absorbing_state, tf.float32),
                action      = tf.cast(next_action, tf.float32),
                reward      = tf.cast(0., tf.float32),
                discount    = tf.cast(1, tf.float32),
                policy_info = (), ## NOTE: just check the notebook
                env_info    = {},
                other_info  = {}
            )
            episode.append(step)

            step = EnvStep(
                step_type   = time_step.StepType.LAST,
                step_num    = tf.cast(t, tf.int64),
                observation = tf.cast(absorbing_state, tf.float32),
                action      = tf.cast(next_action, tf.float32), ## will be ignored because StepType.LAST
                reward      = tf.cast(0., tf.float32),          ## will be ignored because StepType.LAST
                discount    = tf.cast(1, tf.float32),
                policy_info = (), ## NOTE: just check the notebook
                env_info    = {},
                other_info  = {}
            )
            episode.append(step)

            ## update "container"
            episodes.append(episode)
            episode_lengths.append(len(episode))
            episode = []

            ## reset timestep
            t = 0
            step_type = time_step.StepType.FIRST
        
        ## timeout, no need to add absorbing state
        elif t == max_trajectory_length or i == (read_dataset.size - 1):
            ## get action at the terminal state from behavior policy
            ## NOTE: this is hard-coded for standard deviation of 0.3
            with torch.no_grad(): next_action = policy.actor.forward(next_state)
            next_action = next_action + torch.randn_like(next_action) * 0.3

            next_state = torch.cat((next_state, torch.tensor([0.])))
            step = EnvStep(
                step_type   = time_step.StepType.LAST,
                step_num    = tf.cast(t, tf.int64),
                observation = tf.cast(next_state, tf.float32),
                action      = tf.cast(next_action, tf.float32), ## will be ignored because StepType.LAST
                reward      = tf.cast(0., tf.float32),          ## will be ignored because StepType.LAST
                discount    = tf.cast(1, tf.float32),
                policy_info = (), ## NOTE: just check the notebook
                env_info    = {},
                other_info  = {}
            )
            episode.append(step)

            ## update "container"
            episodes.append(episode)
            episode_lengths.append(len(episode))
            episode = []

            ## reset timestep
            t = 0
            step_type = time_step.StepType.FIRST
    
    ## padding shorter `episode` in `episodes`
    max_length = max(episode_lengths)
    for episode in episodes:
        pad_length = max_length - len(episode)
        episode.extend([episode[-1]] * pad_length)
    
    ## this check only valids when there is no terminal condition
    # assert len(episode_lengths) == read_dataset.start_size - 1, "Num episodes do not match."
    # assert max_length == (max_trajectory_length + 1), "We save (s, a, r) and accounts for terminal state."
    # assert sum(episode_lengths) == read_dataset.size + read_dataset.start_size - 1, "Account for terminal state."
    
    batched_episodes = nest_utils.stack_nested_tensors(
       [nest_utils.stack_nested_tensors(episode) for episode in episodes]
    )
    
    ## create the mask which indicates valid timestep on the padded episode
    valid_steps = tf.range(max_length)[None, :] < tf.convert_to_tensor(episode_lengths)[:, None]

    ## write the converted episodes to the `write_dataset`
    add_episodes_to_dataset(
        episodes        = batched_episodes,
        valid_ids       = valid_steps,
        write_dataset   = write_dataset
    )

    ## save the dataset
    print("Saving dataset to %s" % save_path)
    if not tf.io.gfile.isdir(save_path):
        tf.io.gfile.makedirs(save_path)
    write_dataset.save(save_path)

    ## try loading the dataset
    print("Loading the dataset")
    loaded_dataset = Dataset.load(save_path)

    estimate_bysteps = get_fullbatch_average(loaded_dataset)
    estimate_not_bysteps = get_fullbatch_average(loaded_dataset, by_steps=False)

    print(">> Num loaded steps          :", loaded_dataset.num_steps)
    print(">> Num loaded total steps    :", loaded_dataset.num_total_steps)
    print(">> Num loaded episodes       :", loaded_dataset.num_episodes)
    print(">> Num loaded total episodes :", loaded_dataset.num_total_episodes)
    print(">> Per step average          :", estimate_bysteps)
    print(">> Per episode average       :", estimate_not_bysteps)