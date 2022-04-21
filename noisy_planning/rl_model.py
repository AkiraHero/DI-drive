import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union
from ding.model.common.head import DuelingHead, RegressionHead, ReparameterizationHead, MultiHead, DiscreteHead

########## image encode for rl ########

class BEVVehicleStateEncoder(nn.Module):
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
        self._hidden_dim_list = hidden_dim_list
        self._kernel_size = kernel_size
        self._stride = stride

        self._relu = nn.ReLU()

        # self.bev_road_model = self.get_cnn()
        # self.bev_obj_model = self.get_cnn()
        self.state_model = self.get_linear_encoder()

    def get_linear_encoder(self):
        l1 = nn.Linear(287, 256)
        l2 = nn.Linear(256, 256)
        l3 = nn.Linear(256, 128)
        layers = [l1, self._relu, l2, self._relu, l3, self._relu]
        return nn.Sequential(*layers)


    def get_cnn(self):
        layers = []
        input_dim = self._obs_shape[0]
        for i in range(len(self._hidden_dim_list)):
            layers.append(nn.Conv2d(input_dim, self._hidden_dim_list[i], self._kernel_size[i], self._stride[i]))
            layers.append(self._relu)
            if i == 0:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            input_dim = self._hidden_dim_list[i]
        layers.append(nn.Flatten())
        cnn_layers = nn.Sequential(*layers)
        flatten_size = self._get_flatten_size(cnn_layers)
        linear_tail = nn.Linear(flatten_size, 128)
        layers.append(linear_tail)
        return nn.Sequential(*layers)

    def _get_flatten_size(self, m) -> int:
        test_data = torch.randn(1, *self._obs_shape)
        with torch.no_grad():
            output = m(test_data)
        return output.shape[1]

    def forward(self, data: Dict) -> torch.Tensor:
        velocity_local = data['velocity_local']
        acceleration_local = data['acceleration_local']
        heading_diff = data['heading_diff']
        last_steer = data['last_steer']
        collide_wall = data['collide_wall']
        collide_obj = data['collide_obj']
        way_curvature = data['way_curvature']
        laser_beam = data['laser_obs']
        lane_dis_obs = data['lane_dis_obs']
        collide_solid_lane = data['collide_solid_lane']

        # place holder
        obstacle_data = torch.zeros_like(laser_beam, device=laser_beam.device)
        neibor_boxes = data['neibor_boxes']

        if 2 == len(velocity_local.shape):  # fill batch_size dim
            velocity_local = velocity_local.unsqueeze(0)
            acceleration_local = acceleration_local.unsqueeze(0)
            heading_diff = heading_diff.unsqueeze(0)
            last_steer = last_steer.unsqueeze(0)
            collide_wall = collide_wall.unsqueeze(0)
            collide_obj = collide_obj.unsqueeze(0)
            way_curvature = way_curvature.unsqueeze(0)
            laser_beam = laser_beam.unsqueeze(0)
            lane_dis_obs = lane_dis_obs.unsqueeze(0)
            collide_solid_lane = collide_solid_lane.unsqueeze(0)
            obstacle_data = obstacle_data.unsqueeze(0)
            neibor_boxes = neibor_boxes.unsqueeze(0)

        valid_dim = min(obstacle_data.shape[1], neibor_boxes.shape[1])
        obstacle_data[:, :valid_dim, :] = neibor_boxes[:, :valid_dim, :]

        state_vec = torch.cat([velocity_local,
                               acceleration_local,
                               heading_diff,
                               last_steer,
                               collide_solid_lane,
                               collide_obj,
                               way_curvature,
                               lane_dis_obs,
                               obstacle_data,
                               ], dim=1).squeeze(-1) # dim: bs, colume vector
        state_embedding = self.state_model(state_vec)
        return state_embedding


# class BEVVehicleStateEncoder(nn.Module):
#     def __init__(
#             self,
#             obs_shape: Tuple,
#             hidden_dim_list: List,
#             embedding_size: int,
#             kernel_size: List = [8, 4, 3],
#             stride: List = [4, 2, 1],
#     ) -> None:
#         super().__init__()
#         assert len(kernel_size) == len(stride), (kernel_size, stride)
#         self._obs_shape = obs_shape
#         self._embedding_size = embedding_size
#         self._hidden_dim_list = hidden_dim_list
#         self._kernel_size = kernel_size
#         self._stride = stride
#
#         self._relu = nn.ReLU()
#
#         self.bev_road_model = self.get_cnn()
#         self.bev_obj_model = self.get_cnn()
#         self.state_model = self.get_linear_encoder()
#
#     def get_linear_encoder(self):
#         l1 = nn.Linear(42, 128)
#         l2 = nn.Linear(128, 256)
#         l3 = nn.Linear(256, 64)
#         layers = [l1, self._relu, l2, self._relu, l3, self._relu]
#         return nn.Sequential(*layers)
#
#
#     def get_cnn(self):
#         layers = []
#         input_dim = self._obs_shape[0]
#         for i in range(len(self._hidden_dim_list)):
#             layers.append(nn.Conv2d(input_dim, self._hidden_dim_list[i], self._kernel_size[i], self._stride[i]))
#             layers.append(self._relu)
#             if i == 0:
#                 layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
#             input_dim = self._hidden_dim_list[i]
#         layers.append(nn.Flatten())
#         cnn_layers = nn.Sequential(*layers)
#         flatten_size = self._get_flatten_size(cnn_layers)
#         linear_tail = nn.Linear(flatten_size, 128)
#         layers.append(linear_tail)
#         return nn.Sequential(*layers)
#
#     def _get_flatten_size(self, m) -> int:
#         test_data = torch.randn(1, *self._obs_shape)
#         with torch.no_grad():
#             output = m(test_data)
#         return output.shape[1]
#
#     def forward(self, data: Dict) -> torch.Tensor:
#         road_img = data['bev_road']
#         obj_img = data['bev_obj']
#         velocity_local = data['velocity_local']
#         acceleration_local = data['acceleration_local']
#         heading_diff = data['heading_diff']
#         last_steer = data['last_steer']
#         collide_wall = data['collide_wall']
#         collide_obj = data['collide_obj']
#         way_curvature = data['way_curvature']
#         if 3 == len(road_img.shape):  # fill batch_size dim
#             road_img = road_img.unsqueeze(0)
#             obj_img = obj_img.unsqueeze(0)
#             velocity_local = velocity_local.unsqueeze(0)
#             acceleration_local = acceleration_local.unsqueeze(0)
#             heading_diff = heading_diff.unsqueeze(0)
#             last_steer = last_steer.unsqueeze(0)
#             collide_wall = collide_wall.unsqueeze(0)
#             collide_obj = collide_obj.unsqueeze(0)
#             way_curvature = way_curvature.unsqueeze(0)
#         road_img = road_img.permute(0, 3, 1, 2)
#         obj_img = obj_img.permute(0, 3, 1, 2)
#         state_vec = torch.cat([velocity_local,
#                                acceleration_local,
#                                heading_diff,
#                                last_steer,
#                                collide_wall,
#                                collide_obj,
#                                way_curvature], dim=1).squeeze(-1) # dim: bs, colume vector
#         road_embedding = self.bev_road_model(road_img)
#         obj_embedding = self.bev_obj_model(obj_img)
#         state_embedding = self.state_model(state_vec)
#         return torch.cat([state_embedding, road_embedding, obj_embedding], dim=1)



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
        birdview_data = data['birdview']
        speed = data['speed']
        if 3 == len(birdview_data.shape):  # fill batch_size dim
            birdview_data = birdview_data.unsqueeze(0)
            speed = speed.unsqueeze(0)
        image = birdview_data.permute(0, 3, 1, 2)

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


class SACRLModel(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            share_encoder: bool = False,
            encoder_hidden_size_list: List = [64, 128, 256],
            encoder_embedding_size: int = 128,
            twin_critic: bool = False,
            actor_head_hidden_size: int = 512,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 512,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            **kwargs,
    ) -> None:
        super().__init__()

        self._act = nn.ReLU()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self.twin_critic = twin_critic
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.actor_encoder = self.critic_encoder = BEVVehicleStateEncoder(
                self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
            )
        else:
            self.actor_encoder = BEVVehicleStateEncoder(
                self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
            )
            self.critic_encoder = BEVVehicleStateEncoder(
                self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
            )

        self.actor = nn.Sequential(
            nn.Linear(encoder_embedding_size, actor_head_hidden_size), activation,
            ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type='conditioned',
                activation=activation,
                norm_type=norm_type
            )
        )
        self.twin_critic = twin_critic
        if self.twin_critic:
            if self.share_encoder:
                self._twin_encoder = self.actor_encoder
            else:
                self._twin_encoder = BEVVehicleStateEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
            self.critic = nn.ModuleList()
            for _ in range(2):
                self.critic.append(
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
                    )
                )
        else:
            self.critic = nn.Sequential(
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

    def forward(self, inputs, mode=None, **kwargs):
        self.mode = ['compute_actor', 'compute_critic']
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x0 = self.critic_encoder(inputs['obs'])
        x0 = torch.cat([x0, inputs['action']], dim=1)
        if self.twin_critic:
            x1 = self._twin_encoder(inputs['obs'])
            x1 = torch.cat([x1, inputs['action']], dim=1)
            x = [m(xi)['pred'] for m, xi in [(self.critic[0], x0), (self.critic[1], x1)]]
        else:
            x = self.critic(x0)['pred']
        return {'q_value': x}

    def compute_actor(self, inputs) -> Dict[str, torch.Tensor]:
        x = self.actor_encoder(inputs)
        x = self.actor(x)
        return {'logit': [x['mu'], x['sigma']]}


class PPORLModel(nn.Module):
    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, Tuple] = 2,
            share_encoder: bool = True,
            continuous: bool = True,
            encoder_embedding_size: int = 128,
            encoder_hidden_size_list: List = [64, 128, 256],
            actor_head_hidden_size: int = 128,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 128,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            sigma_type: Optional[str] = 'independent',
            bound_type: Optional[str] = 'tanh',
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.encoder = BEVVehicleStateEncoder(
                self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
            )
        else:
            self.actor_encoder = BEVVehicleStateEncoder(
                self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
            )
            self.critic_encoder = BEVVehicleStateEncoder(
                self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
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
                fixed_sigma_value=0.1,
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

    def forward(self, inputs, mode=None, **kwargs):
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
