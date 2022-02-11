import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union
from ding.model.common.head import DuelingHead, RegressionHead, ReparameterizationHead, MultiHead, DiscreteHead

########## image encode for rl ########

class BEVSpeedConvEncoder(nn.Module):
    """
    Convolutional encoder of Bird-eye View image and speed input. It takes a BeV image and a speed scalar as input.
    The BeV image is encoded by a convolutional encoder, to get an embedding feature which is half size of the
    embedding length. Then the speed value is repeated for half embedding length time, and concated to the above
    feature to get a final feature.

    :Arguments:
        - obs_shape (Tuple): BeV image shape.
        - hidden_dim_list (List): Conv encoder hidden layer dimension list.
        - embedding_size (int): Embedding feature dimensions.
        - kernel_size (List, optional): Conv kernel size for each layer. Defaults to [8, 4, 3].
        - stride (List, optional): Conv stride for each layer. Defaults to [4, 2, 1].
    """

    def __init__(
            self,
            obs_shape: Tuple,
            hidden_dim_list: List,
            embedding_size: int,
            kernel_size: List = [8, 4, 3],
            stride: List = [4, 2, 1],
    ) -> None:
        super().__init__()
        assert len(kernel_size) == len(stride), (kernel_size, stride)
        self._obs_shape = obs_shape
        self._embedding_size = embedding_size

        self._relu = nn.ReLU()
        layers = []
        input_dim = obs_shape[0]
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Conv2d(input_dim, hidden_dim_list[i], kernel_size[i], stride[i]))
            #if i == 0:
            #    layers.append(nn.BatchNorm2d(hidden_dim_list[i]))
            layers.append(self._relu)
            if i == 0:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            input_dim = hidden_dim_list[i]
        layers.append(nn.Flatten())
        self._model = nn.Sequential(*layers)
        flatten_size = self._get_flatten_size()
        self._mid = nn.Linear(flatten_size, self._embedding_size // 2)

    def _get_flatten_size(self) -> int:
        test_data = torch.randn(1, *self._obs_shape)
        with torch.no_grad():
            output = self._model(test_data)
        return output.shape[1]

    def forward(self, data: Dict) -> torch.Tensor:
        """
        Forward computation of encoder

        :Arguments:
            - data (Dict): Input data, must contain 'birdview' and 'speed'

        :Returns:
            torch.Tensor: Embedding feature.
        """
        image = data['birdview'].permute(0, 3, 1, 2)
        speed = data['speed']
        x = self._model(image)
        x = self._mid(x)
        speed_embedding_size = self._embedding_size - self._embedding_size // 2
        speed_vec = torch.unsqueeze(speed, 1).repeat(1, speed_embedding_size)
        h = torch.cat((x, speed_vec), dim=1)
        return h


############## RL ###########

class DDPGRLModel(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            share_encoder: bool = False,
            encoder_hidden_size_list: List = [64, 128, 256],
            encoder_embedding_size: int = 512,
            twin_critic: bool = False,
            actor_head_hidden_size: int = 512,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 512,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self.twin_critic = twin_critic
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.actor_encoder = self.critic_encoder = BEVSpeedConvEncoder(
                self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
            )
        else:
            self.actor_encoder = BEVSpeedConvEncoder(
                self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
            )
            self.critic_encoder = BEVSpeedConvEncoder(
                self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
            )

        self.actor_head = nn.Sequential(
            nn.Linear(encoder_embedding_size, actor_head_hidden_size), activation,
            RegressionHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                final_tanh=True,
                activation=activation,
                norm_type=norm_type
            )
        )
        self.twin_critic = twin_critic
        if self.twin_critic:
            if not self.share_encoder:
                self._twin_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
            else:
                self._twin_encoder = self.actor_encoder
            self.critic_head = [
                nn.Sequential(
                    nn.Linear(encoder_embedding_size + self._act_shape, critic_head_hidden_size), activation,
                    RegressionHead(
                        critic_head_hidden_size,
                        1,
                        critic_head_layer_num,
                        final_tanh=False,
                        activation=activation,
                        norm_type=norm_type
                    )
                ) for _ in range(2)
            ]
        else:
            self.critic_head = nn.Sequential(
                nn.Linear(encoder_embedding_size + self._act_shape, critic_head_hidden_size), activation,
                RegressionHead(
                    critic_head_hidden_size,
                    1,
                    critic_head_layer_num,
                    final_tanh=False,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        # for convenience of call some apis(such as: self.critic.parameters()), but may cause
        # misunderstanding when print(self)
        self.actor = nn.ModuleList([self.actor_encoder, self.actor_head])
        if self.twin_critic:
            self.critic = nn.ModuleList([self.critic_encoder, *self.critic_head, self._twin_encoder])
        else:
            self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

    def forward(self, inputs, mode=None, **kwargs):
        assert (mode in ['compute_actor_critic', 'compute_actor', 'compute_critic'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def compute_critic(self, inputs: Dict) -> Dict:
        x0 = self.critic_encoder(inputs['obs'])
        x0 = torch.cat([x0, inputs['action']], dim=1)
        if self.twin_critic:
            x1 = self._twin_encoder(inputs['obs'])
            x1 = torch.cat([x1, inputs['action']], dim=1)
            x = [m(xi)['pred'] for m, xi in [(self.critic_head[0], x0), (self.critic_head[1], x1)]]
        else:
            x = self.critic_head(x0)['pred']
        return {'q_value': x}

    def compute_actor(self, inputs: Dict) -> Dict:
        x = self.actor_encoder(inputs)
        action = self.actor_head(x)['pred']
        return {'action': action}


class TD3RLModel(DDPGRLModel):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            share_encoder: bool = False,
            encoder_hidden_size_list: List = [32, 64, 128],
            encoder_embedding_size: int = 512,
            twin_critic: bool = True,
            actor_head_hidden_size: int = 512,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 512,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        super().__init__(
            obs_shape, action_shape, share_encoder, encoder_hidden_size_list, encoder_embedding_size,
            twin_critic, actor_head_hidden_size, actor_head_layer_num, critic_head_hidden_size,
            critic_head_layer_num, activation, norm_type)
        assert twin_critic


class DQNRLModel(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, Tuple] = 21,
            encoder_hidden_size_list: Tuple = [32, 64, 128],
            dueling: bool = True,
            head_hidden_size: Optional[int] = 512,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        super().__init__()
        self._encoder = BEVSpeedConvEncoder(obs_shape, encoder_hidden_size_list, head_hidden_size, [3, 3, 3], [2, 2, 2])
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self._head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self._head = head_cls(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )

    def forward(self, obs):
        x = self._encoder(obs)
        y = self._head(x)
        return y
