import os
import time
import carla
import numpy as np
from typing import Any, Dict, Optional, Tuple
from gym import spaces
from collections import defaultdict

from .base_drive_env import BaseDriveEnv
from core.simulators import CarlaSimulator
from core.utils.others.visualizer import Visualizer
from core.utils.simulator_utils.carla_utils import visualize_birdview
from core.utils.env_utils.stuck_detector import StuckDetector
from core.utils.simulator_utils.carla_utils import lane_mid_distance, get_lane_dis

import traceback


class SimpleCarlaEnv(BaseDriveEnv):
    """
    A simple deployment of Carla Environment with single hero vehicle. It use ``CarlaSimulator`` to interact with
    Carla server and gets running status. The observation is obtained from simulator's state, information and
    sensor data, along with reward which can be calculated and retrived.

    When created, it will initialize environment with config and Carla TCP host & port. This method will NOT create
    simulator instance. It only creates some data structures to store information when running env.

    :Arguments:
        - cfg (Dict): Env config dict.
        - host (str, optional): Carla server IP host. Defaults to 'localhost'.
        - port (int, optional): Carla server IP port. Defaults to 9000.
        - tm_port (Optional[int], optional): Carla Traffic Manager port. Defaults to None.

    :Interfaces: reset, step, close, is_success, is_failure, render, seed

    :Properties:
        - hero_player (carla.Actor): Hero vehicle in simulator.
    """

    metadata = {'render.modes': ['rgb_array']}
    action_space = spaces.Dict({})
    observation_space = spaces.Dict({})
    reward_type = ['goal', 'distance', 'speed', 'angle', 'steer', 'lane', 'failure']
    config = dict(
        simulator=dict(),
        # reward types total reward take into account
        reward_type=['goal', 'distance', 'speed', 'angle', 'failure'],
        # reward value if success
        success_reward=10,
        failure_reward=-5,
        # failure judgement
        col_is_failure=False,
        stuck_is_failure=False,
        ignore_light=False,
        ran_light_is_failure=False,
        off_road_is_failure=False,
        wrong_direction_is_failure=False,
        off_route_is_failure=False,
        # failure judgement hyper-parameters
        off_route_distance=6,
        success_distance=5,
        stuck_len=300,
        max_speed=5,
        # whether open visualize
        visualize=None,
        reward_func="ini_compute_reward",
    )

    def __init__(
            self,
            cfg: Dict,
            host: str = 'localhost',
            port: int = 9000,
            tm_port: Optional[int] = None,
            carla_timeout: Optional[int] = 60.0,
            **kwargs,
    ) -> None:
        """
        Initialize environment with config and Carla TCP host & port.
        """
        import logging
        self.logger = logging.getLogger("[Simulator-{}]".format(port))
        while self.logger.handlers:
            self.logger.handlers.pop()
        if len(self.logger.handlers) == 0:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(thread)0x- %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        super().__init__(cfg, **kwargs)

        self._simulator_cfg = self._cfg.simulator
        self._carla_host = host
        self._carla_port = port
        self._carla_tm_port = tm_port
        self._carla_timeout = carla_timeout

        self._use_local_carla = False
        if self._carla_host != 'localhost':
            self._use_local_carla = True
        self._simulator = None

        self._col_is_failure = self._cfg.col_is_failure
        self._stuck_is_failure = self._cfg.stuck_is_failure
        self._ignore_light = self._cfg.ignore_light
        self._ran_light_is_failure = self._cfg.ran_light_is_failure
        self._off_road_is_failure = self._cfg.off_road_is_failure
        self._wrong_direction_is_failure = self._cfg.wrong_direction_is_failure
        self._off_route_is_failure = self._cfg.off_route_is_failure
        self._off_route_distance = self._cfg.off_route_distance

        self._reward_type = self._cfg.reward_type
        assert set(self._reward_type).issubset(self.reward_type), set(self._reward_type)
        self._success_distance = self._cfg.success_distance
        self._success_reward = self._cfg.success_reward
        self._failure_reward = self._cfg.failure_reward
        self._max_speed = self._cfg.max_speed
        self._collided = False
        self._stuck = False
        self._ran_light = False
        self._off_road = False
        self._wrong_direction = False
        self._off_route = False
        self._stuck_detector = StuckDetector(self._cfg.stuck_len)

        self._tick = 0
        self._timeout = float('inf')
        self._launched_simulator = False

        self._visualize_cfg = self._cfg.visualize
        self._simulator_databuffer = dict()
        self._visualizer = None
        self._last_canvas = None
        self.logger.warning("Using reward function: {}, suc_reward={}, fail_reward={}".format(self._cfg.reward_func, self._success_reward, self._failure_reward))
        self._reward_func = self.__getattribute__(self._cfg.reward_func)
        self._suite_name = None

    def _init_carla_simulator(self) -> None:
        if not self._use_local_carla:
            print("------ Run Carla on Port: %d, GPU: %d ------" % (self._carla_port, 0))
            #self.carla_process = subprocess.Popen()
            self._simulator = CarlaSimulator(
                cfg=self._simulator_cfg,
                client=None,
                host=self._carla_host,
                port=self._carla_port,
                tm_port=self._carla_tm_port,
                timeout=self._carla_timeout,
            )
        else:
            print('------ Using Remote carla @ {}:{} ------'.format(self._carla_host, self._carla_port))
            self._simulator = CarlaSimulator(
                cfg=self._simulator_cfg,
                client=None,
                host=self._carla_host,
                port=self._carla_port,
                tm_port=self._carla_tm_port
            )
        self._launched_simulator = True

    def reset(self, **kwargs) -> Dict:
        """
        Reset environment to start a new episode, with provided reset params. If there is no simulator, this method will
        create a new simulator instance. The reset param is sent to simulator's ``init`` method to reset simulator,
        then reset all statues recording running states, and create a visualizer if needed. It returns the first frame
        observation.

        :Returns:
            Dict: The initial observation.
        """
        self.logger.error("[Reset paras]{}".format(str(kwargs)))
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
                vis_name = "{}_port{}_{}".format(
                    self._simulator.town_name, self._carla_port, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                )

            self._visualizer.init(vis_name)

        if 'col_is_failure' in kwargs:
            self._col_is_failure = kwargs['col_is_failure']
        if 'stuck_is_failure' in kwargs:
            self._stuck_is_failure = kwargs['stuck_is_failure']
        if 'suite_name' in kwargs:
            self._suite_name = kwargs['suite_name']
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
        """
        Run one time step of environment, get observation from simulator and calculate reward. The environment will
        be set to 'done' only at success or failure. And if so, all visualizers will end. Its interfaces follow
        the standard definition of ``gym.Env``.

        :Arguments:
            - action (Dict): Action provided by policy.

        :Returns:
            Tuple[Any, float, bool, Dict]: A tuple contains observation, reward, done and information.
        """
        # if self.weird_bug:
            # self.logger.error("now print weird bug stacks===========================================")
            # traceback.print_stack()            
            # self.logger.error("============================================hope it helps u============")
        # self.logger.error("start to perform step... ")
        if action is not None:
            for key in ['throttle', 'brake']:
                if key in action:
                    np.clip(action[key], 0, 1)
            if 'steer' in action:
                np.clip(action['steer'], -1, 1)
            self._simulator.apply_control(action)
            self._simulator_databuffer['action'] = action
        else:
            self._simulator_databuffer['action'] = dict()
        # self.logger.error("start to perform step... before simu runstep")
        self._simulator.run_step()
        self._tick += 1
        # self.logger.error("start to perform step... after simu runstep, tick=".format(self._tick))

        obs = self.get_observations()

        self._collided = self._simulator.collided
        self._stuck = self._stuck_detector.stuck
        self._ran_light = self._simulator.ran_light
        self._off_road = self._simulator.off_road
        self._wrong_direction = self._simulator.wrong_direction

        location = self._simulator_databuffer['state']['location'][:2]
        target = self._simulator_databuffer['navigation']['target']
        self._off_route = np.linalg.norm(location - target) >= self._off_route_distance
        self._reward, reward_info = self._reward_func()

        info = self._simulator.get_information()
        info.update(reward_info)
        info.update(
            {
                'collided': self._collided,
                'stuck': self._stuck,
                'ran_light': self._ran_light,
                'off_road': self._off_road,
                'wrong_direction': self._wrong_direction,
                'off_route': self._off_route,
                'timeout': self._tick > self._timeout,
                'success': self.is_success(),
            }
        )
        if self._suite_name:
            info.update({
                'suite_name': self._suite_name
            })
        # self.render()

        done = self.is_success() or self.is_failure()
        # self.logger.error("current step  done={}, tick={}".format(done, self._tick))
        if done:
            # self.logger.error("I double this bug is because 1.1.................................................................")
            # traceback.print_stack()
            # self.weird_bug = True
            self._simulator.clean_up()
            # self.logger.error("I double this bug is because 1.2.................................................................")
            if self._visualizer is not None:
                # self.logger.error("visual this is the 280 line of carla env")
                self._visualizer.done()
                # self.logger.error("visual this is the 282 line of carla env")
                self._visualizer = None
            # self.logger.error("done over.................................................................")
        return obs, self._reward, done, info

    def close(self) -> None:
        """
        Delete simulator & visualizer instances and close the environment.
        """
        if self._launched_simulator:
            # self.logger.error("I double this bug is because 2.1.................................................................")
            self._simulator.clean_up()
            # self.logger.error("I double this bug is because 2.2.................................................................")
            self._simulator._set_sync_mode(False)
            del self._simulator
            self._launched_simulator = False
        if self._visualizer is not None:
            self._visualizer.done()
            self._visualizer = None

    def is_success(self) -> bool:
        """
        Check if the task succeed. It only happens when hero vehicle is close to target waypoint.

        :Returns:
            bool: Whether success.
        """
        if self._simulator.end_distance < self._success_distance:
            return True
        return False

    def is_failure(self) -> bool:
        """
        Check if env fails. colliding, being stuck, running light, running off road, running in
        wrong direction according to config. It will certainly happen when time is out.

        :Returns:
            bool: Whether failure.
        """
        if self._stuck_is_failure and self._stuck:
            return True
        if self._col_is_failure and self._collided:
            return True
        if self._ran_light_is_failure and self._ran_light:
            return True
        if self._off_road_is_failure and self._off_road:
            return True
        if self._wrong_direction_is_failure and self._wrong_direction:
            return True
        if self._off_route_is_failure and self._off_route:
            return True
        if self._tick > self._timeout:
            return True

        return False

    def get_observations(self) -> Dict:
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
        return obs

    # https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/carla_env.py
    def compute_reward_cheat(self):
        out_lane_thres = 2.0
        desired_speed = 8.0

        """Calculate the step reward."""
        # reward for speed tracking
        v = self.hero_player.get_velocity()
        # speed = np.sqrt(v.x ** 2 + v.y ** 2)
        # r_speed = -abs(speed - desired_speed)

        # reward for collision
        r_collision = 0
        if self._collided:
            r_collision = -1

        # reward for steering:
        r_steer = -self.hero_player.get_control().steer ** 2

        # reward for out of lane
        location = self._simulator_databuffer['state']['location']
        ego_x, ego_y = location[0], location[1]
        waypoint_list = self._simulator_databuffer['navigation']['waypoint_list']
        dis, w = get_lane_dis(waypoint_list, ego_x, ego_y)
        r_out = 0
        if abs(dis) > out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > desired_speed:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.hero_player.get_control().steer) * lspeed_lon ** 2
        reward_info = {}
        reward_info['collision_reward'] = 200 * r_collision
        reward_info['speed_reward_longitude'] = 1 * lspeed_lon
        reward_info['speed_reward_fast'] = 10 * r_fast
        reward_info['lane_reward'] = 1 * r_out
        reward_info['steer_reward'] = r_steer * 5
        reward_info['lateral_acc_reward'] = 0.2 * r_lat - 0.1
        total_reward = 0
        for k, v in reward_info.items():
            total_reward += v
        return total_reward, reward_info


    def customized_compute_reward(self) -> Tuple[float, Dict]:
        """
        Compute reward for current frame, with details returned in a dict. In short, in contains goal reward,
        route following reward calculated by route length in current and last frame, some navigation attitude reward
        with respective to target waypoint, and failure reward by checking each failure event.

        :Returns:
            Tuple[float, Dict]: Total reward value and detail for each value.
        """

        def dist(loc1, loc2):
            return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5

        def angle(vec1, vec2):
            cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
            cos = np.clip(cos, -1, 1)
            angle = np.arccos(cos)
            angle = np.clip(angle, 0, np.pi)
            return angle

        # goal reward
        goal_reward = 0
        plan_distance = self._simulator.end_distance
        if self.is_success():
            goal_reward += self._success_reward
        elif self.is_failure():
            goal_reward -= 1

        # distance reward
        location = self._simulator_databuffer['state']['location']
        target = self._simulator_databuffer['navigation']['target']
        target_distance = dist(target, location)
        cur_distance = plan_distance + target_distance
        if self._last_distance is None:
            distance_reward = 0
        else:
            distance_reward = np.clip((self._last_distance - cur_distance) * 2, 0, 1)
        self._last_distance = cur_distance

        # state reward: speed, angle, steer, mid lane
        speed = self._simulator_databuffer['state']['speed'] / 3.6
        speed_limit = self._simulator_databuffer['navigation']['speed_limit'] / 3.6
        speed_limit = min(self._max_speed, speed_limit)
        agent_state = self._simulator_databuffer['navigation']['agent_state']
        target_speed = speed_limit
        if agent_state == 2 or agent_state == 3:
            target_speed = 0
        elif agent_state == 4 and not self._ignore_light:
            target_speed = 0
        # speed_reward = 1 - abs(speed - target_speed) / speed_limit
        speed_reward = 0
        if speed < target_speed / 5:
            speed_reward -= 1
        if speed > target_speed:
            speed_reward -= 1

        forward_vector = self._simulator_databuffer['state']['forward_vector']
        target_forward = self._simulator_databuffer['navigation']['target_forward']
        angle_reward = 3 * (0.1 - angle(forward_vector, target_forward) / np.pi)

        steer = self._simulator_databuffer['action'].get('steer', 0)
        command = self._simulator_databuffer['navigation']['command']
        steer_reward = 0.5
        if abs(steer - self._last_steer) > 0.5:
            steer_reward -= 0.2
        if command == 1 and steer > 0.1:
            steer_reward = 0
        elif command == 2 and steer < -0.1:
            steer_reward = 0
        elif (command == 3 or command == 4) and abs(steer) > 0.3:
            steer_reward = 0
        self._last_steer = steer

        waypoint_list = self._simulator_databuffer['navigation']['waypoint_list']

        # todo: the lane mid dis should be checked carefully in turn area / open road-crossing
        lane_mid_dis = abs(lane_mid_distance(waypoint_list, location))
        lrw = 0.0
        if lane_mid_dis < 0.1:
            lrw = 0.3
        elif lane_mid_dis < 0.3:
            lrw = 0.1
        elif lane_mid_dis < 0.6:
            lrw = 0.0
        elif lane_mid_dis < 0.8:
            lrw = -0.3
        elif lane_mid_dis < 1.0:
            lrw = -0.5
        else:
            lrw = -2.0
        lane_reward = lrw

        failure_reward = 0
        if self._col_is_failure and self._collided:
            failure_reward += self._failure_reward
        elif self._stuck_is_failure and self._stuck:
            failure_reward += self._failure_reward
        elif self._off_road_is_failure and self._off_road:
            failure_reward += self._failure_reward
        elif self._ran_light_is_failure and not self._ignore_light and self._ran_light:
            failure_reward += self._failure_reward
        elif self._wrong_direction_is_failure and self._wrong_direction:
            failure_reward += self._failure_reward
        # add weight to failure reward
        #failure_reward *= 50

        reward_info = {}
        total_reward = 0
        reward_info['goal_reward'] = goal_reward
        reward_info['distance_reward'] = distance_reward
        reward_info['speed_reward'] = speed_reward
        reward_info['angle_reward'] = angle_reward
        reward_info['steer_reward'] = steer_reward
        reward_info['lane_reward'] = lane_reward
        reward_info['failure_reward'] = failure_reward

        reward_dict = {
            'goal': goal_reward,
            'distance': distance_reward,
            'speed': speed_reward,
            'angle': angle_reward,
            'steer': steer_reward,
            'lane': lane_reward,
            'failure': failure_reward
        }
        for rt in self._reward_type:
            total_reward += reward_dict[rt]

        return total_reward, reward_info
    

    def ini_compute_reward(self) -> Tuple[float, Dict]:
        """
        Compute reward for current frame, with details returned in a dict. In short, in contains goal reward,
        route following reward calculated by route length in current and last frame, some navigation attitude reward
        with respective to target waypoint, and failure reward by checking each failure event.

        :Returns:
            Tuple[float, Dict]: Total reward value and detail for each value.
        """

        def dist(loc1, loc2):
            return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5

        def angle(vec1, vec2):
            cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
            cos = np.clip(cos, -1, 1)
            angle = np.arccos(cos)
            angle = np.clip(angle, 0, np.pi)
            return angle

        # goal reward
        goal_reward = 0
        plan_distance = self._simulator.end_distance
        if self.is_success():
            goal_reward += self._success_reward
        elif self.is_failure():
            goal_reward -= 1

        # distance reward
        location = self._simulator_databuffer['state']['location']
        target = self._simulator_databuffer['navigation']['target']
        target_distance = dist(target, location)
        cur_distance = plan_distance + target_distance
        if self._last_distance is None:
            distance_reward = 0
        else:
            distance_reward = np.clip((self._last_distance - cur_distance) * 2, 0, 1)
        self._last_distance = cur_distance

        # state reward: speed, angle, steer, mid lane
        speed = self._simulator_databuffer['state']['speed'] / 3.6
        speed_limit = self._simulator_databuffer['navigation']['speed_limit'] / 3.6
        speed_limit = min(self._max_speed, speed_limit)
        agent_state = self._simulator_databuffer['navigation']['agent_state']
        target_speed = speed_limit
        if agent_state == 2 or agent_state == 3:
            target_speed = 0
        elif agent_state == 4 and not self._ignore_light:
            target_speed = 0
        # speed_reward = 1 - abs(speed - target_speed) / speed_limit
        speed_reward = 0
        if speed < target_speed / 5:
            speed_reward -= 1
        if speed > target_speed:
            speed_reward -= 1

        forward_vector = self._simulator_databuffer['state']['forward_vector']
        target_forward = self._simulator_databuffer['navigation']['target_forward']
        angle_reward = 3 * (0.1 - angle(forward_vector, target_forward) / np.pi)

        steer = self._simulator_databuffer['action'].get('steer', 0)
        command = self._simulator_databuffer['navigation']['command']
        steer_reward = 0.5
        if abs(steer - self._last_steer) > 0.5:
            steer_reward -= 0.2
        if command == 1 and steer > 0.1:
            steer_reward = 0
        elif command == 2 and steer < -0.1:
            steer_reward = 0
        elif (command == 3 or command == 4) and abs(steer) > 0.3:
            steer_reward = 0
        self._last_steer = steer

        waypoint_list = self._simulator_databuffer['navigation']['waypoint_list']
        lane_mid_dis = lane_mid_distance(waypoint_list, location)
        lane_reward = -0.5 * lane_mid_dis

        failure_reward = 0
        if self._col_is_failure and self._collided:
            failure_reward -= 5
        elif self._stuck_is_failure and self._stuck:
            failure_reward -= 5
        elif self._off_road_is_failure and self._off_road:
            failure_reward -= 5
        elif self._ran_light_is_failure and not self._ignore_light and self._ran_light:
            failure_reward -= 5
        elif self._wrong_direction_is_failure and self._wrong_direction:
            failure_reward -= 5

        reward_info = {}
        total_reward = 0
        reward_info['goal_reward'] = goal_reward
        reward_info['distance_reward'] = distance_reward
        reward_info['speed_reward'] = speed_reward
        reward_info['angle_reward'] = angle_reward
        reward_info['steer_reward'] = steer_reward
        reward_info['lane_reward'] = lane_reward
        reward_info['failure_reward'] = failure_reward

        reward_dict = {
            'goal': goal_reward,
            'distance': distance_reward,
            'speed': speed_reward,
            'angle': angle_reward,
            'steer': steer_reward,
            'lane': lane_reward,
            'failure': failure_reward
        }
        for rt in self._reward_type:
            total_reward += reward_dict[rt]

        return total_reward, reward_info

    def render(self, mode='rgb_array') -> Any:
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
        return self._visualizer.canvas

    def seed(self, seed: int) -> None:
        """
        Set random seed for environment.

        :Arguments:
            - seed (int): Random seed value.
        """
        print('[ENV] Setting seed:', seed)
        np.random.seed(seed)

    def __repr__(self) -> str:
        return "SimpleCarlaEnv - host %s : port %s" % (self._carla_host, self._carla_port)

    @property
    def hero_player(self) -> carla.Actor:
        return self._simulator.hero_player
