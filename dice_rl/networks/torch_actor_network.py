"""
Transfer our pretrained TD3 expert from torch to tensorflow.

NOTE: This is very hard-coded. Careful when you try to use it.

How to use:
actor_net = TorchActorNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=(256, 256)
)
actor_net.load_torch_weights(path)
ie. using default `continuous_projection_net` and ignoring the standard deviation.
It is meant to replace the actor creationg part in `get_sac_policy()` in `dice_rl/environments/env_policies.py`.
"""

import numpy as np
import torch

from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork


class TorchActorNetwork(ActorDistributionNetwork):
    """
    Enable load weights from a torch model.
    """

    def load_torch_weights(self, torch_model_path: str, is_pendulum: bool = False) -> None:
        """
        Load weights from a torch model.

        :param torch_model_path: Path to the torch model.
        """
        torch_state_dict = torch.load(torch_model_path, map_location="cpu")

        ## self._encoder.layers = [Flatten(), Dense(), Dense()]
        ## self._projection_network.layers = [Dense(), BiasLayer()]

        ## first dense layer
        weight = torch_state_dict["net.0.weight"].numpy().T
        bias = torch_state_dict["net.0.bias"].numpy()
        self._encoder.layers[1].set_weights([weight, bias])

        ## second dense layer
        weight = torch_state_dict["net.2.weight"].numpy().T
        bias = torch_state_dict["net.2.bias"].numpy()
        self._encoder.layers[2].set_weights([weight, bias])

        ## final dense layer
        if is_pendulum:
            weight = torch_state_dict["net.4.weight"].numpy().T
            dummy_action_weight = np.random.randn(*weight.shape)
            weight = np.concatenate((dummy_action_weight, weight), axis=1)
            
            bias = torch_state_dict["net.4.bias"].numpy()
            dummy_action_bias = np.random.randn(*bias.shape)
            bias = np.concatenate((dummy_action_bias, bias))
            
            self._projection_networks.layers[0].set_weights([weight, bias])

        else:
            weight = torch_state_dict["net.4.weight"].numpy().T
            bias = torch_state_dict["net.4.bias"].numpy()
            self._projection_networks.layers[0].set_weights([weight, bias])

        del torch_state_dict