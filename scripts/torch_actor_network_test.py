import os
import d4rl

from tf_agents.environments import suite_mujoco, tf_py_environment
from tf_agents.policies.actor_policy import ActorPolicy

from dice_rl.networks.torch_actor_network import TorchActorNetwork

## GPU is problematic for current installation of TF on storm servers
os.environ["CUDA_VISIBLE_DEVICES"] = ""

env = suite_mujoco.load("HalfCheetah-v3")
env = tf_py_environment.TFPyEnvironment(suite_mujoco.load("maze2d-umaze-v0"))

actor_net = TorchActorNetwork(
    input_tensor_spec   = env.observation_spec(),
    output_tensor_spec  = env.action_spec(),
    fc_layer_params     = (256, 256)
)

## build the actor_network
## need to build first before load torch weights
## what i mean is try to call `.summary()` before building
policy = ActorPolicy(
    time_step_spec  = env.time_step_spec(),
    action_spec     = env.action_spec(),
    actor_network   = actor_net,
    training        = False
)

print("---------------------------")
print("Network Architecture")
for l in actor_net._encoder.layers:
    print(type(l))
    try:
        print(l.activation)
    except:
        print("No activation")

for l in actor_net._projection_networks.layers:
    print(type(l))
    try:
        print(l.activation)
    except:
        print("No activation")
print("---------------------------")

actor_net.load_torch_weights("scripts/dataset/maze2d-umaze-v0/policies/expert/actor")
print("Done")