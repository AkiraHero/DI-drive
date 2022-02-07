import copy
from core.envs.simple_carla_env import SimpleCarlaEnv

import numpy as np
from typing import Any, Dict, Tuple

from core.utils.others.visualizer import Visualizer
from core.utils.simulator_utils.carla_utils import visualize_birdview
import time

# todo: should use decorator as wrapper... I am lazy..

class SimpleCarlaEnvNewRender(SimpleCarlaEnv):
    def __init__(self, *args, **kwargs):
        super(SimpleCarlaEnvNewRender, self).__init__(*args, **kwargs)
        self._visualizer_dual = None
        self._render_buffer_dual = None

    def reset(self, **kwargs) -> Dict:
        """
        Reset environment to start a new episode, with provided reset params. If there is no simulator, this method will
        create a new simulator instance. The reset param is sent to simulator's ``init`` method to reset simulator,
        then reset all statues recording running states, and create a visualizer if needed. It returns the first frame
        observation.

        :Returns:
            Dict: The initial observation.
        """
        if not self._launched_simulator:
            self._init_carla_simulator()

        self._simulator.init(**kwargs)

        if self._visualize_cfg is not None:
            if self._visualizer is not None:
                self._visualizer.done()
            else:
                self._visualizer = Visualizer(self._visualize_cfg)

            if 'name' in kwargs:
                vis_name = kwargs['name']
            else:
                vis_name = "{}_{}".format(
                    self._simulator.town_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                )

            self._visualizer.init(vis_name)
            ############ dual
            if self._visualizer_dual is not None:
                self._visualizer_dual.done()
            else:
                self._visualizer_dual = Visualizer(self._visualize_cfg)

            if 'name' in kwargs:
                vis_name = kwargs['name']
            else:
                vis_name = "{}_{}".format(
                    self._simulator.town_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                )
            vis_name = vis_name + "_dual"
            self._visualizer_dual.init(vis_name)
            ###############


        if 'col_is_failure' in kwargs:
            self._col_is_failure = kwargs['col_is_failure']
        if 'stuck_is_failure' in kwargs:
            self._stuck_is_failure = kwargs['stuck_is_failure']
        self._simulator_databuffer.clear()
        self._collided = False
        self._stuck = False
        self._ran_light = False
        self._off_road = False
        self._wrong_direction = False
        self._off_route = False
        self._stuck_detector.clear()
        self._tick = 0
        self._reward = 0
        self._last_steer = 0
        self._last_distance = None
        self._timeout = self._simulator.end_timeout

        return self.get_observations()


    def step(self, action: Dict) -> Tuple[Any, float, bool, Dict]:
        res = super().step(action)
        done = self.is_success() or self.is_failure()
        # self.logger.error("current step  done={}, tick={}".format(done, self._tick))
        if done:
            if self._visualizer_dual is not None:
                # self.logger.error("visual this is the 280 line of carla env")
                self._visualizer_dual.done()
                # self.logger.error("visual this is the 282 line of carla env")
                self._visualizer_dual = None
        return res

    def close(self) -> None:
        res = super().close()
        if self._visualizer_dual is not None:
            self._visualizer_dual.done()
            self._visualizer_dual = None
        return res

    # following 2 functions are core
    def get_observations(self):
        """
                Get observations from simulator. The sensor data, navigation, state and information in simulator
                are used, while not all these are added into observation dict.

                :Returns:
                    Dict: Observation dict.
                """
        obs = dict()
        state = self._simulator.get_state()
        navigation = self._simulator.get_navigation()
        sensor_data = self._simulator.get_sensor_data()
        information = self._simulator.get_information()

        self._simulator_databuffer['state'] = state
        self._simulator_databuffer['navigation'] = navigation
        self._simulator_databuffer['information'] = information
        if 'action' not in self._simulator_databuffer:
            self._simulator_databuffer['action'] = dict()
        if not navigation['agent_state'] == 4 or self._ignore_light:
            self._stuck_detector.tick(state['speed'])

        obs.update(sensor_data)
        obs.update(
            {
                'tick': information['tick'],
                'timestamp': np.float32(information['timestamp']),
                'agent_state': navigation['agent_state'],
                'node': navigation['node'],
                'node_forward': navigation['node_forward'],
                'target': np.float32(navigation['target']),
                'target_forward': np.float32(navigation['target_forward']),
                'command': navigation['command'],
                'speed': np.float32(state['speed']),
                'speed_limit': np.float32(navigation['speed_limit']),
                'location': np.float32(state['location']),
                'forward_vector': np.float32(state['forward_vector']),
                'acceleration': np.float32(state['acceleration']),
                'velocity': np.float32(state['velocity']),
                'angular_velocity': np.float32(state['angular_velocity']),
                'rotation': np.float32(state['rotation']),
                'is_junction': np.float32(state['is_junction']),
                'tl_state': state['tl_state'],
                'tl_dis': np.float32(state['tl_dis']),
                'waypoint_list': navigation['waypoint_list'],
                'direction_list': navigation['direction_list'],
            }
        )

        if self._visualizer is not None:
            if self._visualize_cfg.type not in sensor_data:
                raise ValueError("visualize type {} not in sensor data!".format(self._visualize_cfg.type))
            self._render_buffer = sensor_data[self._visualize_cfg.type].copy()
            if self._visualize_cfg.type == 'birdview':
                self._render_buffer = visualize_birdview(self._render_buffer)

        # todo: customize this section
        if self._visualizer_dual is not None:
            if self._visualize_cfg.type not in sensor_data:
                raise ValueError("visualize type {} not in sensor data!".format(self._visualize_cfg.type))
            self._render_buffer_dual = copy.deepcopy(sensor_data[self._visualize_cfg.type])
            # move visualize_birdview to render(), where we can get access to detection/gt result
        return obs

    def render(self, mode='rgb_array', obs_with_det=None) -> Any:
        """
        Render a runtime visualization on screen, save a gif or video according to visualizer config.
        The main canvas is from a specific sensor data. It only works when 'visualize' is set in config dict.

        :Returns:
            Any: visualized canvas, mainly used by tensorboard and gym monitor wrapper
        """
        if self._visualizer is None:
            return self._last_canvas

        render_info = {
            'collided': self._collided,
            'off_road': self._off_road,
            'wrong_direction': self._wrong_direction,
            'off_route': self._off_route,
            'reward': self._reward,
            'tick': self._tick,
            'end_timeout': self._simulator.end_timeout,
            'end_distance': self._simulator.end_distance,
            'total_distance': self._simulator.total_distance,
        }
        render_info.update(self._simulator_databuffer['state'])
        render_info.update(self._simulator_databuffer['navigation'])
        render_info.update(self._simulator_databuffer['information'])
        render_info.update(self._simulator_databuffer['action'])

        self._visualizer.paint(self._render_buffer, render_info)
        self._visualizer.run_visualize()
        self._last_canvas = self._visualizer.canvas

        if obs_with_det is not None:
            ini_gt_vehicle_dim = obs_with_det['ini_vehicle_dim']
            ini_gt_walker_dim = obs_with_det['ini_walker_dim']
            # insert to _render_buffer_dual
            """
            0 road
            1 lane
            2 red light
            3 yellow light
            4 green light
            5 vehicle
            6 pedestrian
            7 hero
            8 route
            """
            self._render_buffer_dual[..., 5] = ini_gt_vehicle_dim
            self._render_buffer_dual[..., 6] = ini_gt_walker_dim

        if self._visualize_cfg.type == 'birdview':
            self._render_buffer_dual = visualize_birdview(self._render_buffer_dual)

        self._visualizer_dual.paint(self._render_buffer_dual, render_info)
        self._visualizer_dual.run_visualize()

        return self._visualizer.canvas