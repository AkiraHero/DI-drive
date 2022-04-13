import cv2
import numpy as np
import carla
import re
import weakref
# no pygame because pygame only allow one in a process
# import pygame
import math
import datetime
import os
from carla import ColorConverter as cc

def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# class FadingText(object):
#     """ Class for fading text """
#
#     def __init__(self, font, dim, pos):
#         """Constructor method"""
#         self.font = font
#         self.dim = dim
#         self.pos = pos
#         self.seconds_left = 0
#         self.surface = np.zeros(dim)
#
#     def set_text(self, text, color=(255, 255, 255), seconds=2.0):
#         """Set fading text"""
#         text_texture = self.font.render(text, True, color)
#         self.surface = np.zeros(self.dim)
#         self.seconds_left = seconds
#         self.surface.fill((0, 0, 0, 0))
#         self.surface.blit(text_texture, (10, 11))
#
#     def tick(self, _, clock):
#         """Fading text method for every tick"""
#         delta_seconds = 1e-3 * clock.get_time()
#         self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
#         self.surface.set_alpha(500.0 * self.seconds_left)
#
#     def render(self, display):
#         """Render fading text method"""
#         display.blit(self.surface, self.pos)

# class HUD(object):
#     """Class for HUD text"""
#
#     def __init__(self, width, height):
#         """Constructor method"""
#         self.dim = (width, height)
#         font = pygame.font.Font(pygame.font.get_default_font(), 20)
#         font_name = 'courier' if os.name == 'nt' else 'mono'
#         fonts = [x for x in pygame.font.get_fonts() if font_name in x]
#         default_font = 'ubuntumono'
#         mono = default_font if default_font in fonts else fonts[0]
#         mono = pygame.font.match_font(mono)
#         self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
#         self._notifications = FadingText(font, (width, 40), (0, height - 40))
#         # self.help = HelpText(pygame.font.Font(mono, 24), width, height)
#         self.server_fps = 0
#         self.frame = 0
#         self.simulation_time = 0
#         self._show_info = True
#         self._info_text = []
#         self._server_clock = pygame.time.Clock()
#
#     def on_world_tick(self, timestamp):
#         """Gets informations from the world at every tick"""
#         self._server_clock.tick()
#         self.server_fps = self._server_clock.get_fps()
#         self.frame = timestamp.frame_count
#         self.simulation_time = timestamp.elapsed_seconds
#
#     def tick(self, world, clock):
#         """HUD method for every tick"""
#         self._notifications.tick(world, clock)
#         if not self._show_info:
#             return
#         transform = world.player.get_transform()
#         vel = world.player.get_velocity()
#         control = world.player.get_control()
#         heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
#         heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
#         heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
#         heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
#         colhist = world.collision_sensor.get_collision_history()
#         collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
#         max_col = max(1.0, max(collision))
#         collision = [x / max_col for x in collision]
#         vehicles = world.world.get_actors().filter('vehicle.*')
#
#         self._info_text = [
#             'Server:  % 16.0f FPS' % self.server_fps,
#             'Client:  % 16.0f FPS' % clock.get_fps(),
#             '',
#             'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
#             'Map:     % 20s' % world.map.name,
#             'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
#             '',
#             'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
#             u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
#             'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
#             'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
#             'Height:  % 18.0f m' % transform.location.z,
#             '']
#         if isinstance(control, carla.VehicleControl):
#             self._info_text += [
#                 ('Throttle:', control.throttle, 0.0, 1.0),
#                 ('Steer:', control.steer, -1.0, 1.0),
#                 ('Brake:', control.brake, 0.0, 1.0),
#                 ('Reverse:', control.reverse),
#                 ('Hand brake:', control.hand_brake),
#                 ('Manual:', control.manual_gear_shift),
#                 'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
#         elif isinstance(control, carla.WalkerControl):
#             self._info_text += [
#                 ('Speed:', control.speed, 0.0, 5.556),
#                 ('Jump:', control.jump)]
#         self._info_text += [
#             '',
#             'Collision:',
#             collision,
#             '',
#             'Number of vehicles: % 8d' % len(vehicles)]
#
#         if len(vehicles) > 1:
#             self._info_text += ['Nearby vehicles:']
#
#         def dist(l):
#             return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
#                              ** 2 + (l.z - transform.location.z)**2)
#         vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]
#
#         for dist, vehicle in sorted(vehicles):
#             if dist > 200.0:
#                 break
#             vehicle_type = get_actor_display_name(vehicle, truncate=22)
#             self._info_text.append('% 4dm %s' % (dist, vehicle_type))
#
#     def toggle_info(self):
#         """Toggle info on or off"""
#         self._show_info = not self._show_info
#
#     def notification(self, text, seconds=2.0):
#         """Notification text"""
#         self._notifications.set_text(text, seconds=seconds)
#
#     def error(self, text):
#         """Error text"""
#         self._notifications.set_text('Error: %s' % text, (255, 0, 0))
#
#     def render(self, display):
#         """Render for HUD class"""
#         if self._show_info:
#             info_surface = pygame.Surface((220, self.dim[1]))
#             info_surface.set_alpha(100)
#             display.blit(info_surface, (0, 0))
#             v_offset = 4
#             bar_h_offset = 100
#             bar_width = 106
#             for item in self._info_text:
#                 if v_offset + 18 > self.dim[1]:
#                     break
#                 if isinstance(item, list):
#                     if len(item) > 1:
#                         points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
#                         pygame.draw.lines(display, (255, 136, 0), False, points, 2)
#                     item = None
#                     v_offset += 18
#                 elif isinstance(item, tuple):
#                     if isinstance(item[1], bool):
#                         rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
#                         pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
#                     else:
#                         rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
#                         pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
#                         fig = (item[1] - item[2]) / (item[3] - item[2])
#                         if item[2] < 0.0:
#                             rect = pygame.Rect(
#                                 (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
#                         else:
#                             rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
#                         pygame.draw.rect(display, (255, 255, 255), rect)
#                     item = item[0]
#                 if item:  # At this point has to be a str.
#                     surface = self._font_mono.render(item, True, (255, 255, 255))
#                     display.blit(surface, (8, v_offset))
#                 v_offset += 18
#         self._notifications.render(display)


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """Constructor method"""
        self.sensor = None
        self.image = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(1280))
                blp.set_attribute('image_size_y', str(720))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None
        self.parse_img_cnt = 0
        self.rendered = False

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.image = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.image is not None:
            if self.rendered:
                print("There is no new image come in........")

            display.fill(0)
            display += self.image
            self.rendered = True

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        self.parse_img_cnt += 1
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.image = lidar_img
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            # print("{}:Here is image settimg..................................................".format(self.parse_img_cnt))
            self.image = array
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)
        self.rendered = False


class CarlaActorProbeVisualizer(object):
    def __init__(self, actor):

        self._width, self._height = 1280, 720
        self._img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._actor = actor
        # self._hud = HUD(1280, 720)
        self._hud = None
        self._camera_gamma = 2.2
        self._camera_manager = CameraManager(self._actor, self._hud, self._camera_gamma)
        self._cam_pos_id = 0
        self._cam_index = 0
        self._camera_manager.transform_index = self._cam_pos_id
        self._camera_manager.set_sensor(self._cam_index, notify=False)
        pass

    def destroy_sensors(self):
        """Destroy sensors"""
        self._camera_manager.sensor.destroy()
        self._camera_manager.sensor = None
        self._camera_manager.index = None

    def render(self, birdview=None):
        self._camera_manager.render(self._img)
        if birdview is not None:
            self.add_birdview(birdview)
        # self._hud.render(self._img)

    def get_visualize_img(self, birdview=None):
        self.render(birdview)
        return self._img

    def reset(self, actor):
        self._img.fill(0)
        self._actor = actor
        self.destroy_sensors()
        self._camera_manager = CameraManager(self._actor, self._hud, self._camera_gamma)
        self._camera_manager.transform_index = self._cam_pos_id
        self._camera_manager.set_sensor(self._cam_index, notify=False)


    def add_birdview(self, birdview):
        birdview_img = self.get_birdview_obs_image(birdview)
        h = birdview_img.shape[0]
        w = birdview_img.shape[0]
        self._img[:h, :w, ...] = birdview_img

    def get_birdview_obs_image(self, birdview):
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

        def render_birdview_image(bird_view_data_dict):
            BACKGROUND = [0, 0, 0]
            bev_render_colors = {
                'road': (85, 87, 83),
                'lane': (211, 215, 207),
                'lightRed': (255, 0, 0),
                'lightYellow': (255, 255, 0),
                'lightGreen': (0, 255, 0),
                'vehicle': (252, 175, 62),
                'pedestrian': (173, 74, 168),
                'hero': (32, 74, 207),
                'route': (41, 239, 41),
            }
            canvas = None
            for k, v in bird_view_data_dict.items():
                if canvas is None:
                    h, w = v.shape
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    canvas[...] = BACKGROUND
                canvas[v > 0.5] = bev_render_colors[k]
            return canvas

        chn_dict = {
            'road': birdview[..., 0],
            'lane': birdview[..., 1],
            'vehicle': birdview[..., 5],
            'pedestrian': birdview[..., 6],
            'route': birdview[..., 8],
            'hero': birdview[..., 7],
        }

        render_buffer = render_birdview_image(chn_dict)
        return render_buffer