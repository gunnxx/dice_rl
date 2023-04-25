"""
Convert our custom `ReplayBuffer` to `TFOffpolicyDataset`.
"""

import argparse
import d4rl
import numpy as np
import os.path
import pickle
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
from replay_buffer import ReplayBuffer


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
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    ## TODO
    load_path = args.load_path
    filename  = load_path.split("/")[-1]
    env_name  = load_path.split("/")[-3]
    save_path = os.path.join(args.save_path, filename[:-3])

    # assert env_name in [
    #     "Ant-v3",
    #     "HalfCheetah-v3",
    #     "Hopper-v3",
    #     "Humanoid-v3",
    #     "Walker2d-v3"
    # ], "All these environments have maximum length of 1000."

    ## load our dataset
    # with open(load_path, "rb") as f:
    #     read_dataset : ReplayBuffer = pickle.load(f)
    read_dataset : ReplayBuffer = torch.load(load_path, map_location=torch.device("cpu"))

    num_trajectory = read_dataset.start_size
    max_trajectory_length = 150

    ## load environment to create spec
    ## NOTE: use `suite_mujoco` instead of `suite_gym` because it converts every `gym.spaces.Box` to `float32`
    env = suite_mujoco.load(env_name)
    env = tf_py_environment.TFPyEnvironment(env)

    ## create the environment spec
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
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
        capacity = num_trajectory * (max_trajectory_length + 1)
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

        ## convert to `EnvStep`
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

        ## terminal state or maximum length or end of buffer
        if not not_done or \
           t == max_trajectory_length or \
           i == (read_dataset.size - 1):
            ## add the terminal state
            step = EnvStep(
                step_type   = time_step.StepType.LAST,
                step_num    = tf.cast(t, tf.int64),
                observation = tf.cast(next_state, tf.float32),
                action      = tf.cast(action, tf.float32),
                reward      = tf.cast(reward, tf.float32),
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
    
    assert len(episode_lengths) == read_dataset.start_size - 1, "Num episodes do not match."
    assert max_length == (max_trajectory_length + 1), "We save (s, a, r) and accounts for terminal state."
    assert sum(episode_lengths) == read_dataset.size + read_dataset.start_size - 1, "Account for terminal state."
    
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