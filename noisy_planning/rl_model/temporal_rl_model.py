import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union
from ding.model.common.head import DuelingHead, RegressionHead, ReparameterizationHead, MultiHead, DiscreteHead


# backbone

class ChannelEncoder(nn.Module):
    def __init__(self,
                 obs_shape,
                 hidden_dim_list=[32, 64, 128],
                 kernel_size=[3, 3, 3],
                 stride=[2, 2, 2],
                 embedding_size=256,
                 ):
        super().__init__()
        self._obs_shape = obs_shape
        self._embedding_size = embedding_size
        layers = []
        input_dim = obs_shape[0]
        self._relu = nn.ReLU()
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Conv2d(input_dim, hidden_dim_list[i], kernel_size[i], stride[i]))
            layers.append(self._relu)
            if i == 0:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            input_dim = hidden_dim_list[i]
        layers.append(nn.Flatten())
        self._model = nn.Sequential(*layers)
        flatten_size = self._get_flatten_size()
        self._mid = nn.Linear(flatten_size, self._embedding_size)

    def _get_flatten_size(self) -> int:
        test_data = torch.randn(1, *self._obs_shape)
        with torch.no_grad():
            output = self._model(test_data)
        return output.shape[1]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = self._model(data)
        x = self._mid(x)
        return x


class BEVTemporalSpeedConvEncoder(nn.Module):
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
            obs_seq_len: int,
            embedding_size: int,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._obs_seq_len = obs_seq_len
        self._embedding_size = embedding_size

        self._channel_num = obs_shape[0]
        channel_obs_shape = [1, *(obs_shape[1:])]
        self._channel_encoder_embedding_size = 256
        self._motion_encoding_dim = self._channel_encoder_embedding_size
        self._temporal_hidder_num = 64
        self._channel_encoder = nn.ModuleList([ChannelEncoder(obs_shape=channel_obs_shape,
                                                embedding_size=self._channel_encoder_embedding_size)
                                 for i in range(self._channel_num)])
        self._temporal_unit = nn.ModuleList([nn.GRU(self._channel_encoder_embedding_size + self._motion_encoding_dim,
                                      self._temporal_hidder_num, batch_first=True)
                               for i in range(self._channel_num)])
        self._final = nn.Linear(self._temporal_hidder_num * self._channel_num * self._obs_seq_len, self._embedding_size)

    def forward(self, data: Dict) -> torch.Tensor:
        """
        Forward computation of encoder

        :Arguments:
            - data (Dict): Input data, must contain 'birdview' and 'speed'

        :Returns:
            torch.Tensor: Embedding feature.
        """
        chn_temporal_embedding_list = []
        for chn_inx in range(self._channel_num):
            chn_embedding_steps = []
            for timestep in data[::-1]:  # reverse it
                batch_channels = timestep['birdview']
                speed = timestep['speed']
                chn = batch_channels[:, chn_inx: chn_inx + 1, :, :]
                chn_embedding = self._channel_encoder[chn_inx](chn)
                # involve control information: speed
                motion_encoding = torch.unsqueeze(speed, 1).repeat(1, self._motion_encoding_dim)
                chn_embedding = torch.cat([chn_embedding, motion_encoding], dim=1)
                chn_embedding_steps.append(chn_embedding)
            # chn_embedding_steps dim: seq, batch, feature
            chn_embedding_steps = torch.stack(chn_embedding_steps, dim=1)
            temporal_embedding, _ = self._temporal_unit[chn_inx](chn_embedding_steps)
            temporal_embedding = nn.Flatten()(temporal_embedding)
            chn_temporal_embedding_list.append(temporal_embedding)
        embedding_total = torch.cat(chn_temporal_embedding_list, dim=1)
        return self._final(embedding_total)


class PPOTemporalRLModel(nn.Module):
    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            obs_seq_len: int = 1,
            action_shape: Union[int, Tuple] = 2,
            share_encoder: bool = True,
            continuous: bool = True,
            encoder_embedding_size: int = 256,
            actor_head_hidden_size: int = 256,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 256,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            sigma_type: Optional[str] = 'independent',
            bound_type: Optional[str] = None,

    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._obs_seq_len = obs_seq_len
        self._act_shape = action_shape
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.encoder = BEVTemporalSpeedConvEncoder(
                self._obs_shape, obs_seq_len, encoder_embedding_size,
            )
        else:
            self.actor_encoder = BEVTemporalSpeedConvEncoder(
                self._obs_shape, obs_seq_len, encoder_embedding_size,
            )
            self.critic_encoder = BEVTemporalSpeedConvEncoder(
                self._obs_shape, obs_seq_len, encoder_embedding_size,
            )
        self.critic_head = RegressionHead(
            critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
        )
        self.continuous = continuous
        if self.continuous:
            self.multi_head = False
            self.actor_head = ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type
            )
        else:
            multi_head = not isinstance(action_shape, int)
            self.multi_head = multi_head
            if multi_head:
                self.actor_head = MultiHead(
                    DiscreteHead,
                    actor_head_hidden_size,
                    action_shape,
                    layer_num=actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            else:
                self.actor_head = DiscreteHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
        # for convenience of call some apis(such as: self.critic.parameters()), but may cause
        # misunderstanding when print(self)
        if self.share_encoder:
            self.actor = nn.ModuleList([self.encoder, self.actor_head])
            self.critic = nn.ModuleList([self.encoder, self.critic_head])
        else:
            self.actor = nn.ModuleList([self.actor_encoder, self.actor_head])
            self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

    def preprocess_data_batch(self, data: List):
        assert isinstance(data, list)
        timesteps = len(data)
        assert timesteps == self._obs_seq_len
        for d in data:
            if 3 == len(d['birdview'].shape):  # fill batch_size dim
                d['birdview'] = d['birdview'].unsqueeze(0)
                d['speed'] = d['speed'].unsqueeze(0)
            d['birdview'] = d['birdview'].permute(0, 3, 1, 2)
        obs_shape = data[0]['birdview'].shape[1:]
        assert obs_shape == torch.Size(self._obs_shape)
        return data

    def forward(self, inputs, mode=None, **kwargs):
        inputs = self.preprocess_data_batch(inputs)
        assert (mode in ['compute_actor_critic', 'compute_actor', 'compute_critic'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def compute_actor_critic(self, inputs) -> Dict[str, torch.Tensor]:
        if self.share_encoder:
            actor_embedding = critic_embedding = self.encoder(inputs)
        else:
            actor_embedding = self.actor_encoder(inputs)
            critic_embedding = self.critic_encoder(inputs)
        value = self.critic_head(critic_embedding)
        actor_output = self.actor_head(actor_embedding)
        if self.continuous:
            logit = [actor_output['mu'], actor_output['sigma']]
        else:
            logit = actor_output['logit']
        return {'logit': logit, 'value': value['pred']}

    def compute_actor(self, inputs: Dict) -> Dict:
        if self.share_encoder:
            x = self.encoder(inputs)
        else:
            x = self.actor_encoder(inputs)
        x = self.actor_head(x)
        if self.continuous:
            x = {'logit': [x['mu'], x['sigma']]}
        return x

    def compute_critic(self, inputs: Dict) -> Dict:
        if self.share_encoder:
            x = self.encoder(inputs)
        else:
            x = self.critic_encoder(inputs)
        x = self.critic_head(x)
        return {'value': x['pred']}
