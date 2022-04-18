'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description: Carla simulator.
'''
import numpy as np
import random
from typing import Optional, Dict
from collections import defaultdict

from .base_simulator import BaseSimulator
from core.utils.simulator_utils.sensor_utils import SensorHelper, CollisionSensor, TrafficLightHelper
from core.utils.simulator_utils.map_utils import BeVWrapper
from core.simulators.carla_data_provider import CarlaDataProvider
from core.utils.simulator_utils.carla_utils import control_to_signal, get_birdview
from core.utils.planner import BasicPlanner, BehaviorPlanner, LBCPlannerNew
from core.utils.others.tcp_helper import find_traffic_manager_port

import carla
from carla import WeatherParameters

from noisy_planning.utils.debug_utils import generate_general_logger

PRESET_WEATHERS = {
    1: WeatherParameters.ClearNoon,
    2: WeatherParameters.CloudyNoon,
    3: WeatherParameters.WetNoon,
    4: WeatherParameters.WetCloudyNoon,
    5: WeatherParameters.MidRainyNoon,
    6: WeatherParameters.HardRainNoon,
    7: WeatherParameters.SoftRainNoon,
    8: WeatherParameters.ClearSunset,
    9: WeatherParameters.CloudySunset,
    10: WeatherParameters.WetSunset,
    11: WeatherParameters.WetCloudySunset,
    12: WeatherParameters.MidRainSunset,
    13: WeatherParameters.HardRainSunset,
    14: WeatherParameters.SoftRainSunset,
}

VEHICLE_NAME = 'vehicle.tesla.model3'
ROLE_NAME = 'hero'
OBS_TYPE_LIST = ['state', 'depth', 'rgb', 'segmentation', 'bev', 'lidar', 'gnss']

PLANNER_DICT = {
    'basic': BasicPlanner,
    'behavior': BehaviorPlanner,
    'lbc': LBCPlannerNew,
}


class CarlaSimulator(BaseSimulator):
    """
    Common Carla Simulator.
    The simulator creates a client to Carla server, and is able to get observation, send
    control signals to the hero vehicle and record essential data from the simulated world.
    In the initialization period, the simulator may change the environment parameters including
    maps and weathers and can add actors (including NPC vehicles, pedestrians as well as sensors
    mounted on the hero vehicle),
    During the running period the simulator will achieve running state and information about
    the hero vehicle (such as running speed, angle, navigation goal and reference path), data
    from the sensors (such as camera images, lidar points) as well as running status(including
    collision, running off road, red light, distance and timeout to end waypoint).

    Once it is created, it will set up Carla client and set the parameters in the configuration
    dict as its default. When actually calling the ``init`` method to start an episode, some of
    the configurations may be changed by the input arguments while others remain by default.

    The simulator stores and gets some information from a static class ``CarlaDataProvider``
    to avoid frequently sending message to Carla server and speed up.k

    Up to now, it uses Carla version 0.9.9.

    If no traffic manager port is provided, it will find random free port in system.

    :Arguments:
        - cfg (Dict): Config Dict.
        - client (carla.Client, optional): Already established Carla client. Defaults to None.
        - host (str, optional): TCP host Carla client link to. Defaults to 'localhost'.
        - port (int, optional): TCP port Carla client link to. Defaults to 9000.
        - tm_port (int, optional): Traffic manager port Carla client link to. Defaults to None.
        - timeout (float, optional): Carla client link timeout. Defaults to 60.0.

    :Interfaces:
        init, get_state, get_sensor_data, get_navigation, get_information, apply_planner,
        apply_control, run_step, clean_up

    :Properties:
        - town_name (str): Current town name.
        - hero_player (carla.Actor): hero actor in simulation.
        - collided (bool): Whether collided in current episode.
        - ran_light (bool): Whether ran light in current frame.
        - off_road (bool): Whether ran off road in current frame.
        - wrong_direction (bool): Whether ran in wrong direction in current frame.
        - end_distance (float): Distance to target in current frame.
        - end_timeout (float): Timeout for entire route provided by planner.
        - total_distance (float): Distance for entire route provided by planner.
    """
    config = dict(
        town='Town01',
        weather='random',
        sync_mode=True,
        delta_seconds=0.1,
        no_rendering=False,
        auto_pilot=False,
        n_vehicles=0,
        n_pedestrians=0,
        disable_two_wheels=False,
        col_threshold=400,
        waypoint_num=20,
        obs=list(),
        planner=dict(),
        aug=None,
        verbose=True,
        debug=False,
        spawn_manner="random",  # random, near
        spawn_pos_fix=None
    )

    def __init__(
            self,
            cfg: Dict,
            client: Optional[carla.Client] = None,
            host: str = 'localhost',
            port: int = 9000,
            tm_port: Optional[int] = None,
            timeout: float = 60.0,
            **kwargs
    ) -> None:
        logger_name = "[Simulator-{}]".format(port)
        self.logger = generate_general_logger(logger_name)
        """
        Init Carla simulator.
        """
        super().__init__(cfg)
        # omit this procedure
        # Check Carla API version
        # dist = pkg_resources.get_distribution("carla")
        # if LooseVersion(dist.version) < LooseVersion('0.9.8'):
        #     raise ImportError("CARLA version 0.9.8 or newer required. CARLA version found: {}".format(dist))

        # Create the client that will send the requests to the simulator
        if client is None:
            self._client = carla.Client(host, port)
        else:
            self._client = client

        self._client_timeout = timeout
        self._client.set_timeout(self._client_timeout)
        if tm_port is None:
            self.logger.debug("[SIMULATOR] Not providing TM port, try finding free")
            max_retry = 0
            while True:
                try:
                    tm_port = find_traffic_manager_port()
                    self._tm = self._client.get_trafficmanager(tm_port)
                    self.logger.debug("[SIMULATOR] Using TM port:{}".format(tm_port))
                    break
                except Exception as e:
                    max_retry += 1
                    if max_retry > 10:
                        raise e
        else:
            self._tm = self._client.get_trafficmanager(tm_port)
        self._tm.set_global_distance_to_leading_vehicle(2.0)
        self._tm.set_hybrid_physics_mode(True)
        self._world = None
        self._map = None

        self._sync_mode = self._cfg.sync_mode
        self._no_rendering = self._cfg.no_rendering
        self._delta_seconds = self._cfg.delta_seconds
        self._apply_world_setting(**self._cfg)
        self._col_threshold = self._cfg.col_threshold
        self._waypoint_num = self._cfg.waypoint_num
        self._obs_cfg = self._cfg.obs
        self._planner_cfg = self._cfg.planner
        self._camera_aug_cfg = self._cfg.aug
        self._verbose = self._cfg.verbose
        self._spawn_manner = self._cfg.spawn_manner
        self._spawn_pos_path = self._cfg.spawn_pos_fix
        self._spawn_pos_list = None
        if self._spawn_pos_path:
            self._spw_dynamic, self._spawn_pos_list = self._load_spw_pos()


        self._support_spawn_manner = ['random', 'near']
        assert self._spawn_manner in self._support_spawn_manner

        self._tick = 0
        self._timestamp = 0
        self._end_location = None
        self._collided = False
        self._collided_info = None
        self._ran_light = False
        self._off_road = False
        self._wrong_direction = False
        self._end_distance = float('inf')
        self._end_timeout = float('inf')

        self._hero_actor = None
        self._start_location = None
        self._sensor_helper = None
        self._bev_wrapper = None
        self._planner = None
        self._collision_sensor = None
        self._traffic_light_helper = None

        self._start_spawn_point_inx = None
        self._end_spawn_point_inx = None


        self._actor_map = defaultdict(list)
        self._debug = self._cfg.debug



    def _apply_world_setting(self, **world_param) -> None:
        for k in world_param:
            if k is None:
                world_param.pop(k)
        if 'town' in world_param:
            self._town_name = world_param['town']
        if 'weather' in world_param:
            self._weather = world_param['weather']
        if 'n_vehicles' in world_param:
            self._n_vehicles = world_param['n_vehicles']
        if 'n_pedestrians' in world_param:
            self._n_pedestrians = world_param['n_pedestrians']
        if 'autopilot' in world_param:
            self._autopilot = world_param['autopilot']
        if 'disable_two_wheels' in world_param:
            self._disable_two_wheels = world_param['disable_two_wheels']

    def init(self, start: int = 0, end: int = 1, **kwargs) -> None:
        """
        Init simulator episode with provided args.
        This method takes start and end waypoint indexes to set a navigation goal, and will use planner to build a route
        to generate target waypoint and road options in each tick. It will set world, map, vehicles, pedestrians dut to
        default config, and provided args, which will be stored to replace old config.
        If no collision happens when creating actors, the init will end and return.

        :Arguments:
            - start (int, optional): Index of start waypoint. Defaults to 0.
            - end (int, optional): Index of end waypoint. Defaults to 1.

        :Optional arguments: town, weather, n_vehicles, n_pedestrians, autopilot, disable_two_wheels
        """
        self._start_spawn_point_inx = start
        self._end_spawn_point_inx = end
        self._apply_world_setting(**kwargs)
        self._set_town(self._town_name)
        self._set_weather(self._weather)

        self._blueprints = self._world.get_blueprint_library()

        while True:
            self.clean_up()

            CarlaDataProvider.set_client(self._client)
            CarlaDataProvider.set_world(self._world)
            CarlaDataProvider.set_traffic_manager_port(self._tm.get_port())

            self._spawn_hero_vehicle(start_pos=start)
            self._prepare_observations()
            if self._spawn_pos_list is not None:
                self._spawn_vehicles_fixed(self._spawn_pos_list, dynamic=self._spw_dynamic)
            else:
                self._spawn_vehicles()
            self._spawn_pedestrians()

            CarlaDataProvider.on_carla_tick()
            self.apply_planner(end)

            self._collided = False
            self._ran_light = False
            self._off_road = False
            self._wrong_direction = False

            if self._ready():
                if self._debug:
                    self._count_actors()

                break

    def _set_town(self, town: str) -> None:
        if self._world is None:
            self._world = self._client.load_world(town)
        elif self._map.name != town:
            self._world = self._client.load_world(town)
        else:
            self._world = self._client.get_world()
        if self._world.get_snapshot().timestamp.frame > 1e6:
            self._world = self._client.load_world(town)
        self._map = self._world.get_map()
        self._set_sync_mode(self._sync_mode, self._delta_seconds)

    def _set_sync_mode(self, sync: bool, delta_seconds: float = 0.1) -> None:
        settings = self._world.get_settings()
        if settings.synchronous_mode is not sync:
            settings.synchronous_mode = sync
            settings.fixed_delta_seconds = delta_seconds
            settings.no_rendering_mode = self._no_rendering
            self._world.apply_settings(settings)
            #self._tm.set_synchronous_mode(True)

    def _set_weather(self, weather_string):
        if self._verbose:
            self.logger.debug('[SIMULATOR] Setting weather: ' + str(weather_string))

        if weather_string == 'random':
            weather = np.random.choice(list(PRESET_WEATHERS.values()))
        else:
            weather = PRESET_WEATHERS[weather_string]

        self._world.set_weather(weather)

    def _spawn_hero_vehicle(self, start_pos: int = 0) -> None:
        start_waypoint = CarlaDataProvider.get_spawn_point(start_pos)
        self._start_location = start_waypoint.location
        self._hero_actor = CarlaDataProvider.request_new_actor(VEHICLE_NAME, start_waypoint, ROLE_NAME)

    def _sort_spawn_point(self, points: list, st_pt_inx, ed_pt_inx=None):
        # use Manhattan distance (to do use planner distance, which can be calculated and saved before
        st_loc = CarlaDataProvider.get_spawn_point(st_pt_inx).location
        sort_func = lambda pt: abs(pt.location.x - st_loc.x) + abs(pt.location.y - st_loc.y)
        ed_loc = None
        if ed_pt_inx:
            ed_loc = CarlaDataProvider.get_spawn_point(ed_pt_inx).location
            sort_func = lambda pt: abs(pt.location.x - st_loc.x) + abs(pt.location.y - st_loc.y)\
                                   + abs(pt.location.x - ed_loc.x) + abs(pt.location.y - ed_loc.y)

        points.sort(key=sort_func)
        pass
        # todo: if crowded, add random swap

    def _load_spw_pos(self):
        pos_list = []
        with open(self._spawn_pos_path, 'r') as f:
            dynamic = f.readline().strip('\n')
            lines = f.readlines()
            for i in lines:
                pose = [float(num) for num in i.strip('\n').split(' ')]
                pos_list.append(pose)
            if dynamic == "dynamic":
                dynamic = True
            elif dynamic == 'static':
                dynamic = False
            else:
                raise NotImplementedError
            return dynamic, pos_list


    def _spawn_vehicles_fixed(self, poses, dynamic=False) -> None:
        blueprints = self._blueprints.filter('vehicle.*')
        if self._disable_two_wheels:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(poses):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            # spawn the cars and set their autopilot and light state all together
            if dynamic:
                blueprint.set_attribute('role_name', 'autopilot')
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, self._tm.get_port())))
            else:
                blueprint.set_attribute('role_name', 'static')
                batch.append(SpawnActor(blueprint, transform))

        for response in self._client.apply_batch_sync(batch, True):
            if response.error:
                if self._verbose:
                    self.logger.error('[SIMULATOR]' + str(response.error))
            else:
                CarlaDataProvider.register_actor(self._world.get_actor(response.actor_id))


    def _spawn_vehicles(self) -> None:
        blueprints = self._blueprints.filter('vehicle.*')
        if self._disable_two_wheels:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        spawn_points = self._map.get_spawn_points()
        if self._spawn_manner == 'random':
            random.shuffle(spawn_points)
        elif self._spawn_manner == 'near':
            # sort the spawn point near the st / end
            self._sort_spawn_point(spawn_points, self._start_spawn_point_inx, self._end_spawn_point_inx)
        else:
            raise NotImplementedError

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self._n_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, self._tm.get_port())))

        for response in self._client.apply_batch_sync(batch, True):
            if response.error:
                if self._verbose:
                    self.logger.error('[SIMULATOR]' + str(response.error))
            else:
                CarlaDataProvider.register_actor(self._world.get_actor(response.actor_id))

    def _spawn_pedestrians(self) -> None:
        blueprints = self._blueprints.filter('walker.pedestrian.*')
        SpawnActor = carla.command.SpawnActor

        pedestrians_running = 0.5  # how many pedestrians will run
        pedestrians_crossing = 0.8

        peds_spawned = 0
        walkers = []
        controllers = []
        walker_speed = []
        while peds_spawned < self._n_pedestrians:
            spawn_points = []
            _walkers = []
            _controllers = []
            _walker_speed = []

            # 1. take all the random locations to spawn
            while len(spawn_points) < self._n_pedestrians - peds_spawned:
                spawn_point = carla.Transform()
                loc = self._world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point.location = loc
                    if self._spawn_manner == 'random':
                        spawn_points.append(spawn_point)
                    elif self._spawn_manner == 'near':
                        st_loc = CarlaDataProvider.get_spawn_point(self._start_spawn_point_inx).location
                        ed_loc = CarlaDataProvider.get_spawn_point(self._end_spawn_point_inx).location
                        radius = (abs(st_loc.x - ed_loc.x) + abs(st_loc.y - ed_loc.y)) / 2
                        dis = min(abs(spawn_point.location.x - st_loc.x) + abs(spawn_point.location.y - st_loc.y),
                                  abs(spawn_point.location.x - ed_loc.x) + abs(spawn_point.location.y - ed_loc.y))
                        if dis < radius:
                            spawn_points.append(spawn_point)
                    else:
                        raise NotImplementedError

            # 2. spawn the walker object
            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprints)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    if random.random() > pedestrians_running:
                        # walking
                        _walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                    else:
                        # running
                        _walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    if self._verbose:
                        self.logger.error("[SIMULATOR] Walker has no speed")
                    _walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))


            results = self._client.apply_batch_sync(batch, True)


            _walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    if self._verbose:
                        self.logger.error('[SIMULATOR] Walker ' + str(results[i].error))
                else:
                    peds_spawned += 1
                    _walkers.append(results[i].actor_id)
                    _walker_speed2.append(_walker_speed[i])
            _walker_speed = _walker_speed2


            # 3. spawn the walker controller
            walker_controller_bp = self._blueprints.find('controller.ai.walker')
            batch = [SpawnActor(walker_controller_bp, carla.Transform(), walker) for walker in _walkers]

            for result in self._client.apply_batch_sync(batch, True):
                if result.error:
                    if self._verbose:
                        self.logger.error('[SIMULATOR] Walker controller ' + str(result.error))
                else:
                    _controllers.append(result.actor_id)


            # 4. add peds and controllers into actor dict
            controllers.extend(_controllers)
            walkers.extend(_walkers)
            walker_speed.extend(_walker_speed)



        CarlaDataProvider.register_actors(self._world.get_actors(walkers))
        # CarlaDataProvider.register_actors(self._world.get_actors(controllers))


        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self._world.tick()


        # 5. initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self._world.set_pedestrians_cross_factor(pedestrians_crossing)

        for i, controller_id in enumerate(controllers):
            controller = self._world.get_actor(controller_id)
            #controller = CarlaDataProvider.get_actor_by_id(controller_id)
            controller.start()
            controller.go_to_location(self._world.get_random_location_from_navigation())
            controller.set_max_speed(float(walker_speed[i]))
            self._actor_map['walker_controller'].append(controller)


        # example of how to use parameters
        self._tm.global_percentage_speed_difference(30.0)

        self._world.tick()


    def _prepare_observations(self) -> None:
        self._sensor_helper = SensorHelper(self._obs_cfg, self._camera_aug_cfg)
        self._sensor_helper.setup_sensors(self._world, self._hero_actor)
        while not self._sensor_helper.all_sensors_ready():
            self._world.tick()

        for obs_item in self._obs_cfg:
            if obs_item.type == 'bev':
                self._bev_wrapper = BeVWrapper(obs_item)
                self._bev_wrapper.init(self._client, self._world, self._map, self._hero_actor)
        planner_cls = PLANNER_DICT[self._planner_cfg.get('type', 'basic')]
        self._planner = planner_cls(self._planner_cfg)
        self._collision_sensor = CollisionSensor(self._hero_actor, self._col_threshold)
        self._traffic_light_helper = TrafficLightHelper(self._hero_actor)

    def _ready(self, ticks: int = 30) -> bool:
        for _ in range(ticks):
            self.run_step()
            self.get_state()

        self._tick = 0
        self._timestamp = 0

        return not self._collided

    def _count_actors(self) -> None:
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []
        sensors = []
        others = []
        controller = []

        actors = self._world.get_actors()
        actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]

        for actor_with_transform in actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif 'walker.pedestrian' in actor.type_id:
                walkers.append(actor_with_transform)
            elif 'controller.ai.walker' in actor.type_id:
                controller.append(actor_with_transform)
            elif 'sensor' in actor.type_id:
                sensors.append(actor_with_transform)
            else:
                others.append(actor_with_transform)

        beginning = "\n>>>>>>>>>>>>>>>>>>>>>>>\n"
        vehicle_info = "Vehicles:{}\n".format(len(vehicles))
        sensor_info = "Sensors:{}\n".format(len(sensors))
        for ss in sensors:
            sensor_info + '\t' + str(ss[0]) + "\n"
        controller_info = "Controllers:{}\n".format(len(controller))
        walker_info = "Walkers:{}\n".format(len(walkers))
        others_info = "Others:{}\n".format(len(others))
        ending = "<<<<<<<<<<<<<<<<<<<<<<<<"
        out_string = beginning + vehicle_info + sensor_info + controller_info + walker_info + others_info + ending
        self.logger.debug(out_string)
        # for veh in vehicles:
            # print('\t', veh[0].id, veh[0].type_id, veh[0].attributes['role_name'])


    def apply_planner(self, end_idx: int) -> Dict:
        """
        Apply goal waypoint to planner in simulator. The start point is set to current hero vehicle waypoint.

        :Arguments:
            - end_idx (int): Index of end waypoint.

        :Returns:
            Dict: [description]
        """
        assert self._start_location is not None
        self._end_location = CarlaDataProvider.get_spawn_point(end_idx).location
        self._planner.set_destination(self._start_location, self._end_location, clean=True)
        self._total_distance = self._planner.distance_to_goal
        self._end_timeout = self._planner.timeout

    def get_state(self) -> Dict:
        """
        Get running state from current world. It contains location, orientation, speed, acc,
        and the state of surrounding road info such as traffic light and junction.

        :Returns:
            Dict: State dict.
        """
        speed = CarlaDataProvider.get_speed(self._hero_actor) * 3.6
        transform = CarlaDataProvider.get_transform(self._hero_actor)
        location = transform.location
        forward_vector = transform.get_forward_vector()
        up_vector = transform.get_up_vector()
        right_vector = transform.get_right_vector()

        acceleration = CarlaDataProvider.get_acceleration(self._hero_actor)
        angular_velocity = CarlaDataProvider.get_angular_velocity(self._hero_actor)
        velocity = CarlaDataProvider.get_speed_vector(self._hero_actor)

        light_state = self._traffic_light_helper.active_light_state.value
        drive_waypoint = self._map.get_waypoint(
            location,
            project_to_road=False,
        )
        is_junction = False
        if drive_waypoint is not None:
            is_junction = drive_waypoint.is_junction
            self._off_road = False
        else:
            self._off_road = True

        lane_waypoint = self._map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_location = lane_waypoint.transform.location
        lane_forward_vector = lane_waypoint.transform.rotation.get_forward_vector()



        v_forward = velocity.x * forward_vector.x + velocity.y * forward_vector.y + velocity.z * forward_vector.z
        v_up = velocity.x * up_vector.x + velocity.y * up_vector.y + velocity.z * up_vector.z
        v_right = velocity.x * right_vector.x + velocity.y * right_vector.y + velocity.z * right_vector.z
        a_forward = acceleration.x * forward_vector.x + acceleration.y * forward_vector.y + acceleration.z * forward_vector.z
        a_up = acceleration.x * up_vector.x + acceleration.y * up_vector.y + acceleration.z * up_vector.z
        a_right = acceleration.x * right_vector.x + acceleration.y * right_vector.y + acceleration.z * right_vector.z
        state = {
            'speed': speed,
            'location': np.array([location.x, location.y, location.z]),
            'forward_vector': np.array([forward_vector.x, forward_vector.y, forward_vector.z]),
            'acceleration': np.array([acceleration.x, acceleration.y, acceleration.z]),
            'velocity': np.array([velocity.x, velocity.y, velocity.z]),
            'acceleration_local': np.array([a_forward, a_right, a_up]),
            'velocity_local': np.array([v_forward, v_right, v_up]),
            'angular_velocity': np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z]),
            'rotation': np.array([transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll]),
            'is_junction': is_junction,
            'lane_location': np.array([lane_location.x, lane_location.y]),
            'lane_forward': np.array([lane_forward_vector.x, lane_forward_vector.y]),
            'tl_state': light_state,
            'tl_dis': self._traffic_light_helper.active_light_dis,
            'yaw': transform.rotation.yaw,
        }
        if lane_waypoint is None:
            state['lane_forward'] = None
        else:
            lane_forward_vector = lane_waypoint.transform.get_forward_vector()
            state['lane_forward'] = np.array([lane_forward_vector.x, lane_forward_vector.y])

        return state

    def get_sensor_data(self) -> Dict:
        """
        Get all sensor data and bird-eye view data if exist in current world. Bird-eye view will be
        converted to an multi-channel image.

        :Returns:
            Dict: Sensor and Bev data dict.
        """
        sensor_data = self._sensor_helper.get_sensors_data()

        for obs_item in self._obs_cfg:
            if obs_item.type not in OBS_TYPE_LIST:
                raise NotImplementedError("observation type %s not implemented" % obs_item.type)
            elif obs_item.type == 'bev':
                key = obs_item.name
                sensor_data.update({key: get_birdview(self._bev_wrapper.get_bev_data())})
                #sensor_data.update({key + "_initial_dict": self._bev_wrapper.get_bev_data()})
        return sensor_data

    def get_information(self) -> Dict:
        """
        Get running information including time and ran light counts in current world.

        :Returns:
            Dict: Information dict.
        """
        information = {
            'tick': self._tick,
            'timestamp': self._timestamp,
            'total_lights': self._traffic_light_helper.total_lights,
            'total_lights_ran': self._traffic_light_helper.total_lights_ran
        }

        return information

    def get_navigation(self) -> Dict:
        """
        Get navigation info in current world. Most of the contains come from planner.

        :Returns:
            Dict: Navigation dict.
        """
        command = self._planner.node_road_option
        node_location = self._planner.node_waypoint.transform.location
        node_forward = self._planner.node_waypoint.transform.rotation.get_forward_vector()
        target_location = self._planner.target_waypoint.transform.location
        target_forward = self._planner.target_waypoint.transform.rotation.get_forward_vector()
        waypoint_list = self._planner.get_waypoints_list(self._waypoint_num)
        direction_list = self._planner.get_direction_list(self._waypoint_num)
        agent_state = self._planner.agent_state
        speed_limit = self._planner.speed_limit
        self._end_distance = self._planner.distance_to_goal
        self._end_timeout = self._planner.timeout

        if self._bev_wrapper is not None:
            self._bev_wrapper.update_waypoints(waypoint_list)

        waypoint_location_list = []
        for wp in waypoint_list:
            wp_loc = wp.transform.location
            wp_vec = wp.transform.rotation.get_forward_vector()
            waypoint_location_list.append([wp_loc.x, wp_loc.y, wp_vec.x, wp_vec.y])
        waypoint_curvature_list = []
        for inx in range(len(waypoint_list) - 1):
            wp = waypoint_list[inx]
            wp_nxt = waypoint_list[inx + 1]

            loc_diff = wp.transform.location - wp_nxt.transform.location
            wp_dis = (loc_diff.x ** 2 + loc_diff.y ** 2 + loc_diff.z ** 2)
            assert wp_dis > 0
            wp_dis = wp_dis ** 0.5
            node_yaw = wp.transform.rotation.yaw
            node_yaw_nxt = wp_nxt.transform.rotation.yaw
            wp_angle = node_yaw_nxt - node_yaw
            while wp_angle < -180.0:
                wp_angle += 360.0
            while wp_angle > 180.0:
                wp_angle -= 360.0
            curvature = wp_angle / wp_dis
            waypoint_curvature_list.append(curvature)
        while len(waypoint_curvature_list) < self._waypoint_num:
            waypoint_curvature_list.append(0.0)

        # get surround vehicle waypoint inx

        surround_v_info = self.get_neighbour_vehicles_nearest_waypoint_inx(20.0)

        if not self._off_road:
            current_waypoint = self._planner.current_waypoint
            node_waypoint = self._planner.node_waypoint

            # Lanes and roads are too chaotic at junctions
            if current_waypoint.is_junction or node_waypoint.is_junction:
                self._wrong_direction = False
            else:
                node_yaw = node_waypoint.transform.rotation.yaw % 360
                cur_yaw = current_waypoint.transform.rotation.yaw % 360

                wp_angle = (node_yaw - cur_yaw) % 360

                if 150 <= wp_angle <= (360 - 150):
                    self._wrong_direction = True
                else:
                    # Changing to a lane with the same direction
                    self._wrong_direction = False

        navigation = {
            'agent_state': agent_state.value,
            'command': command.value,
            'node': np.array([node_location.x, node_location.y]),
            'node_forward': np.array([node_forward.x, node_forward.y, node_forward.z]),
            'node_yaw': np.array(self._planner.node_waypoint.transform.rotation.yaw),
            'target': np.array([target_location.x, target_location.y, target_location.z]),
            'target_forward': np.array([target_forward.x, target_forward.y, target_forward.z]),
            'waypoint_list': np.array(waypoint_location_list),
            'waypoint_curvature': np.array(waypoint_curvature_list),
            'speed_limit': np.array(speed_limit),
            'direction_list': np.array(direction_list),
            'surround_v_info': surround_v_info
        }
        return navigation

    def run_step(self) -> None:
        """
        Run one step simulation.
        This will tick Carla world and update information for all sensors and measurement.
        """
        self._world.tick()
        self._tick += 1
        world_snapshot = self._world.get_snapshot()
        self._timestamp = world_snapshot.timestamp.elapsed_seconds

        CarlaDataProvider.on_carla_tick()

        if self._planner is not None:
            self._planner.run_step()

        self._collided = self._collision_sensor.collided
        if self._collided:
            self._collided_info = (self._collision_sensor.collided_obj_id, self._collision_sensor.collided_obj_typeid)
        else:
            self._collided_info = None
        self._traffic_light_helper.tick()
        self._ran_light = self._traffic_light_helper.ran_light

        if self._bev_wrapper is not None:
            if CarlaDataProvider._hero_vehicle_route is not None:
                self._bev_wrapper.tick()

    def apply_control(self, control: Dict = None) -> None:
        """
        Apply control signal for hero player in simulator.
        This will send message to the client and the control takes effect in next tick

        :Arguments:
            - control (dict, optional): Control signal dict. Default to None.

        """
        if control is not None:
            control_signal = control_to_signal(control)
            self._hero_actor.apply_control(control_signal)


    def clean_all_manual_actors(self):
        detroy_command = carla.command.DestroyActor
        not_success = True
        try_destroy_cnt = 0
        while not_success and try_destroy_cnt < 2:
            actors =  self._world.get_actors()
            actor_list = [i for i in actors]
            vehicles = []
            walkers = []
            controllers = []
            sensors = []
            for inx, actor in enumerate(actor_list):
                if 'vehicle' in actor.type_id:
                    vehicles.append(actor)
                elif 'walker.pedestrian' in actor.type_id:
                    walkers.append(actor)
                elif 'controller.ai.walker' in actor.type_id:
                    controllers.append(actor)
                elif 'sensor' in actor.type_id:
                    sensors.append(actor)
            destroy_list = []
            ordered_actors = controllers + walkers + sensors + vehicles
            for i in ordered_actors:
                destroy_list.append(detroy_command(i))
            cur_loop_fail_flag = False
            for response in self._client.apply_batch_sync(destroy_list, True):
                if response.error:
                    cur_loop_fail_flag = True
                    if self._verbose:
                        self.logger.error('[SIMULATOR]' + str(response.error))
                    else:
                        break
            if cur_loop_fail_flag:
                self.logger.debug("[SIMULATOR] fail to clean all actors.")
            not_success = cur_loop_fail_flag
            try_destroy_cnt += 1

        


    def clean_up(self) -> None:
        """
        Destroy all actors and sensors in current world. Clear all messages saved in simulator and data provider.
        This will NOT destroy the Carla client, so simulator can use same carla client to start next episode.
        """
        if self._sensor_helper is not None:
            self._sensor_helper.clean_up()
        if self._collision_sensor is not None:
            self._collision_sensor.clear()

        for actor in self._actor_map['walker_controller']:
            if actor.is_alive:
                actor.stop()
                actor.destroy()
        self._actor_map['walker_controller'].clear()
        self._actor_map.clear()

        if self._bev_wrapper is not None:
            self._bev_wrapper.clear()
        if self._planner is not None:
            self._planner.clean_up()
        self._tick = 0
        self._timestamp = 0
        self._collided = False
        self._ran_light = False
        self._off_road = False
        self._wrong_direction = False
        self._end_distance = float('inf')
        self._end_timeout = float('inf')
        CarlaDataProvider.clean_up()
        if self._debug:
            self.logger.debug("After CarlaDataProvider.clean_up()")
            self._count_actors()
            
        # clear all object
        self.clean_all_manual_actors()

    def get_neighbour_vehicles_nearest_waypoint_inx(self, radius=20.0):
        res_dict = {} # id: pos
        actors = CarlaDataProvider.get_actors()
        hero_actor = CarlaDataProvider.get_hero_actor()
        hero_trans = CarlaDataProvider.get_transform(hero_actor)
        for actor in actors:
            if "vehicle" in actor.type_id:
                trans = CarlaDataProvider.get_transform(actor)
                if trans:
                    if ((trans.location.x - hero_trans.location.x) ** 2 + (trans.location.y - hero_trans.location.y)) ** 0.5 < radius:
                        waypoint_inx, = self.get_nearest_waypoint_on_route(trans)
                        res_dict[actor.id] = waypoint_inx
        if hero_actor.id in res_dict.keys():
            res_dict.pop(hero_actor.id)
        return res_dict

    def get_nearest_waypoint_on_route(self, pos):
        waypoints = CarlaDataProvider.get_hero_vehicle_route()
        x = pos.x
        y = pos.y
        dis_min = 1000
        target_inx = 0
        for inx, pt in enumerate(waypoints):
            d = np.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
            if d < dis_min:
                dis_min = d
                target_inx = inx
        return target_inx, waypoints[target_inx]


    @property
    def town_name(self) -> str:
        return self._town_name

    @property
    def hero_player(self) -> carla.Actor:
        return self._hero_actor

    @property
    def collided(self) -> bool:
        return self._collided

    @property
    def collided_info(self) -> tuple:
        return self._collided_info

    @property
    def ran_light(self) -> bool:
        return self._ran_light

    @property
    def off_road(self) -> bool:
        return self._off_road

    @property
    def wrong_direction(self) -> bool:
        return self._wrong_direction

    @property
    def end_distance(self) -> float:
        return self._end_distance

    @property
    def end_timeout(self) -> float:
        return self._end_timeout

    @property
    def total_distance(self) -> float:
        return self._total_distance
