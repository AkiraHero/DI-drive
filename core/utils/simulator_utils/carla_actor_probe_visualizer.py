import copy
import numpy as np
import carla
import re
import weakref
# no pygame because pygame only allow one in a process
# import pygame
import math
import datetime
import os
import collections
from carla import ColorConverter as cc

from PIL import ImageFont, ImageDraw, Image
# todo: sustitute all opencv by pil

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

class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

    def destroy(self):
        self.sensor.destroy()



class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

    def destroy(self):
        self.sensor.destroy()

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
    
    def destroy(self):
        self.sensor.destroy()


class HUD(object):
    """Class for HUD text"""

    def __init__(self, parent_actor):
        """Constructor method"""
        # self.dim = (width, height)
        # font = pygame.font.Font(pygame.font.get_default_font(), 20)
        # font_name = 'courier' if os.name == 'nt' else 'mono'
        # fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        # default_font = 'ubuntumono'
        # mono = default_font if default_font in fonts else fonts[0]
        # mono = pygame.font.match_font(mono)
        # self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        # self._notifications = FadingText(font, (width, 40), (0, height - 40))
        # self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self._actor = parent_actor
        self._world = self._actor.get_world()
        self._map = self._world.get_map()
        self._world.on_tick(self.on_world_tick)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._font_file = os.path.join(os.path.dirname(__file__), "font/ubuntu-mono/UbuntuMono-Regular.ttf")
        self._font_size = 12
        self._font = ImageFont.truetype(self._font_file, self._font_size)
        # self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        # self._server_clock.tick()
        # self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, gnss_sensor=None, collision_sensor=None):
        """HUD method for every tick"""
        # self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = self._actor.get_transform()
        vel = self._actor.get_velocity()
        control = self._actor.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        if collision_sensor:
            colhist = collision_sensor.get_collision_history()
            collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
            max_col = max(1.0, max(collision))
            collision = [x / max_col for x in collision]
        else:
            collision = "Unavailable"

        vehicles = self._world.get_actors().filter('vehicle.*')
        map_name = self._map.name if self._map else "Unavailable"
        GNSS_info = 'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (gnss_sensor.lat, gnss_sensor.lon)) if gnss_sensor else "GNSS: Unavailable"

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            '',
            'Vehicle: % 20s' % get_actor_display_name(self._actor, truncate=20),
            'Map:     % 20s' % map_name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            GNSS_info,
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != self._actor.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        pass
        # self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        pass
        # self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            w, h = display.size
            draw = ImageDraw.Draw(display, "RGBA")
            draw.rectangle(((0, 0), (220, h)), fill=(0, 0, 0, 128))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > h:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(int(x + 8), int(v_offset + 8 + (1 - y) * 30)) for x, y in enumerate(item)]
                        for i in range(len(points) - 1):
                            pt1 = points[i]
                            pt2 = points[i + 1]
                            draw.line((pt1, pt2), fill=(255, 136, 0, 255))
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect_w, rect_h = 6, 6
                        left, top = bar_h_offset, v_offset + 8
                        color = (255, 255, 255, 255)
                        draw.rectangle(((left, top), (left + rect_w, top + rect_h)), fill=color)
                    else:
                        rect_w, rect_h = bar_width, 6
                        left, top = bar_h_offset, v_offset + 8
                        color = (255, 255, 255, 255)
                        draw.rectangle(((left, top), (left + rect_w, top + rect_h)), outline=color)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect_w, rect_h = 6, 6
                            left, top = bar_h_offset + int(fig * (bar_width - 6)), v_offset + 8
                        else:
                            rect_w, rect_h = int(fig * bar_width), 6
                            left, top = bar_h_offset, v_offset + 8
                        draw.rectangle(((left, top), (left + rect_w, top + rect_h)), fill=color)
                    item = item[0]
                if item:  # At this point has to be a str.
                    draw.text((8, v_offset), item, font=self._font, fill=(255, 255, 255, 255))
                v_offset += 18
            # cv2.imshow("display", display)
            # cv2.waitKey(-1)
        # self._notifications.render(display)


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
        assert isinstance(display, Image.Image)
        if self.image is not None:
            if self.rendered:
                print("There is no new image come in........")
            camera_image = Image.fromarray(self.image)
            camera_image.resize(display.size)
            display.paste(camera_image)
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
        self._actor = None
        self._hud = None
        self._camera_manager = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self._width, self._height = 1280, 720

        # self._img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._img = Image.new("RGB", (self._width, self._height))

        # arrange sub figure
        self._subfigure_width = 160
        self._subfigure_rect =  {
            'birdview': ((self._width - self._subfigure_width, 0), (self._width, self._subfigure_width)),
            'linelidar': ((self._width - self._subfigure_width, self._subfigure_width), (self._width, self._subfigure_width * 2)),
        }


        self._camera_gamma = 2.2
        self._cam_pos_id = 0
        self._cam_index = 0
        self.reset(actor)

    def destroy_sensors(self):
        """Destroy sensors"""
        if self._camera_manager:
            if self._camera_manager.sensor is not None:
                self._camera_manager.sensor.destroy()
                self._camera_manager.sensor = None
            self._camera_manager.index = None

        if self.gnss_sensor:
            self.gnss_sensor.destroy()

        if self.collision_sensor:
            self.collision_sensor.destroy()

    def render(self, birdview=None, linelidar=None):
        self._camera_manager.render(self._img)
        self._hud.tick(collision_sensor=self.collision_sensor, gnss_sensor=self.gnss_sensor)
        self._hud.render(self._img)

        if birdview is not None:
            self.add_birdview(birdview)
        if linelidar is not None:
            self.add_line_lidar(linelidar)

    def get_visualize_img(self, birdview=None, linelidar=None):
        self.render(birdview=birdview, linelidar=linelidar)
        return np.array(self._img)

    def reset(self, actor):
        self._img = Image.new("RGB", (self._width, self._height))
        self._actor = actor
        self._hud = HUD(self._actor)
        self.destroy_sensors()
        self._camera_manager = CameraManager(self._actor, self._hud, self._camera_gamma)
        self._camera_manager.transform_index = self._cam_pos_id
        self._camera_manager.set_sensor(self._cam_index, notify=False)
        self.collision_sensor = CollisionSensor(self._actor, self._hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self._actor, self._hud)
        self.gnss_sensor = GnssSensor(self._actor)


    def add_birdview(self, birdview):
        birdview_img = self.get_birdview_obs_image(birdview)
        self._img.paste(Image.fromarray(birdview_img), self._subfigure_rect['birdview'][0])
        # self._img[:h, -w:, ...] = birdview_img

    def add_line_lidar(self, points):
        points = points[:, :3]
        pixel_size = 0.2
        cx = 40
        cy = 80
        img_width = 160
        img_height = 160
        img = Image.new('RGB', (img_width, img_height))
        draw = ImageDraw.Draw(img, "RGBA")
        pt_num = points.shape[0]
        valid_pt_num = 0
        for i in range(pt_num):
            x = points[i, 0]
            y = points[i, 1]
            # to adapt to carla ordinate: exchange x and y
            py = int(x / pixel_size + cx)
            px = int(y / pixel_size + cy)
            range_ = (x ** 2 + y ** 2) ** 0.5
            theta = np.arctan2(y, x)
            if abs(theta) > np.pi / 2.0 or abs(y) < 0.0001:
                continue
            if py < 0 or py >= img_height or px < 0 or px >= img_width:
                continue
            valid_pt_num += 1
            draw.point((px, py), fill=(255, 0, 255, 255))
        py = int(cx)
        px = int(cy)
        draw.ellipse(((px - 3, py - 3), (px + 3, py + 3)), fill=(255, 0, 0, 255))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        self._img.paste(img, self._subfigure_rect['linelidar'][0])

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