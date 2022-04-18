import pickle
import sys
import cv2
import numpy as np
sys.path.append("/home/akira/carla-0.9.11-py3.7-linux-x86_64.egg")
import carla
from core.utils.planner import BasicPlanner, BehaviorPlanner, LBCPlannerNew
from core.simulators.carla_data_provider import CarlaDataProvider

class MapImage(object):
    def __init__(self, carla_world, carla_map, pixels_per_meter=10, load_map_img=None):
        CarlaDataProvider.set_world(world)
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0
        self._line_width = 2
        if self._pixels_per_meter < 3:
            self._line_width = 1

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_lane_surface = self.big_map_surface = np.zeros((width_in_pixels, width_in_pixels, 3))
        # self.big_lane_surface = np.zeros((width_in_pixels, width_in_pixels))
        if load_map_img is None:
            self.draw_road_map(
                self.big_map_surface, self.big_lane_surface, carla_world, carla_map, self.world_to_pixel,
                self.world_to_pixel_width
            )
        else:
            self.big_map_surface = cv2.imread(load_map_img, cv2.IMREAD_COLOR)
        self.map_surface = self.big_map_surface
        self.lane_surface = self.big_lane_surface
        self.planner = BehaviorPlanner(BehaviorPlanner.config)

    def world_to_pixel_width(self, width):
        return int(self.scale * self._pixels_per_meter * width)

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self._pixels_per_meter * \
            (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * \
            (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def draw_road_map(self, map_surface, lane_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        # map_surface.fill(COLOR_ALUMINIUM_4)
        map_surface *= 0.0
        precision = 0.05

        def draw_lane_arrow(surface, points):
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 8 == 0]
            for line in broken_lines:
                for i in range(len(line) - 1):
                    cv2.arrowedLine(surface, line[i], line[i + 1], (0, 255, 255), 1, 8, 0, 5)

        def draw_lane_id(surface, points, road_id):
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 13 == 0]
            for line in broken_lines:
                cv2.putText(surface, str(road_id), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        def draw_lane_marking(surface, points, solid=True):
            if solid:
                # pygame.draw.lines(surface, COLOR_ORANGE_0, False, points, 2)
                for i in range(len(points) - 1):
                    cv2.line(surface, points[i], points[i + 1], (0, 255, 0), self._line_width)
            else:
                broken_lines = [x for n, x in enumerate(zip(*(iter(points), ) * 20)) if n % 3 == 0]
                for line in broken_lines:
                    # pygame.draw.lines(surface, COLOR_ORANGE_0, False, line, 2)
                    for i in range(len(line) - 1):
                        cv2.line(surface, line[i], line[i + 1], (255, 0, 0), self._line_width)

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def does_cross_solid_line(waypoint, shift):
            w = carla_map.get_waypoint(lateral_shift(waypoint.transform, shift), project_to_road=False)
            if w is None or w.road_id != waypoint.road_id:
                return True
            else:
                return (w.lane_id * waypoint.lane_id < 0) or w.lane_id == waypoint.lane_id

        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)

        # draw road surface
        for idx, waypoint in enumerate(topology):
            print("waypts:", idx)
            waypoints = [waypoint]
            nxt = waypoint.next(precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)[0]

            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

            polygon = left_marking + [x for x in reversed(right_marking)]
            polygon = [world_to_pixel(x) for x in polygon]

            if len(polygon) > 2:
                ps = np.array(polygon)
                cv2.fillPoly(map_surface,  [ps], (255, 255, 255))


        # draw lane markers
        for idx, waypoint in enumerate(topology):
            print("waypts:", idx)
            waypoints = [waypoint]
            nxt = waypoint.next(precision)[0]
            cur_road_id = waypoint.road_id
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)[0]

            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            center_marking = [lateral_shift(w.transform, 0) for w in waypoints]

            if 1:#not waypoint.is_intersection:
                if len(left_marking) == 1:
                    continue
                sample = waypoints[int(len(waypoints) / 2)]
                draw_lane_marking(
                    lane_surface, [world_to_pixel(x) for x in left_marking],
                    does_cross_solid_line(sample, -sample.lane_width * 1.1)
                )
                draw_lane_marking(
                    lane_surface, [world_to_pixel(x) for x in right_marking],
                    does_cross_solid_line(sample, sample.lane_width * 1.1)
                )
            if not waypoint.is_intersection:
                draw_lane_arrow(
                    lane_surface, [world_to_pixel(x) for x in center_marking]
                )
                draw_lane_id(
                    lane_surface, [world_to_pixel(x) for x in center_marking], cur_road_id
                )

    def get_route(self, st_point, ed_point):
        self.planner.set_destination(st_point.location, ed_point.location, clean=True)
        return self.planner._route

    def draw_route(self, route):
        route_distance = 0
        mile_stones = [100 * i for i in range(20)]
        mile_stone_inx = 0
        for i in range(len(route) - 1):
            w1 = route[i]
            w2 = route[i + 1]
            pt1 = self.world_to_pixel(w1[0].transform.location)
            pt2 = self.world_to_pixel(w2[0].transform.location)
            cv2.line(self.map_surface, pt1, pt2, (0, 0, 255), 5)

            delta = w1[0].transform.location - w2[0].transform.location
            dis_ = (delta.x ** 2 + delta.y ** 2) ** 0.5
            route_distance += dis_
            if route_distance > mile_stones[mile_stone_inx]:
                pt = self.world_to_pixel(w2[0].transform.location)
                cv2.circle(self.map_surface, pt, 10, (0, 0, 255), 20)
                cv2.putText(self.map_surface, str(mile_stones[mile_stone_inx]) + "m", [pt[0] - 12, pt[1]], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2)
                print(str(mile_stones[mile_stone_inx]) + "m", w2[0].transform)
                mile_stone_inx += 1

    def draw_spw_pts(self, spw_pts):
        for inx, i in enumerate(spw_pts):
            pt = self.world_to_pixel(i.location)
            cv2.circle(self.map_surface, pt, 10, (0, 255, 0), 20)

        for inx, i in enumerate(spw_pts):
            pt = self.world_to_pixel(i.location)
            cv2.putText(self.map_surface, str(inx), [pt[0] - 12, pt[1]], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 26, 253), 2)



def save_spw_pt(pts, filename):
    infos = []
    for i in pts:
        info = [i.location.x, i.location.y, i.location.z, i.rotation.roll, i.rotation.pitch,i.rotation.yaw]
        infos.append(info)
    with open(filename, 'wb') as f:
        pickle.dump(infos, f)

if __name__ == '__main__':
    client = carla.Client("localhost", 9000)
    town = 'TOWN05'
    world = client.load_world(town)
    map = world.get_map()
    map_image = MapImage(world, map, load_map_img="/home/akira/Project/Model_behaviour/DI-drive/noisy_planning/utils/map_town5.png")
    print("_world_offset", map_image._world_offset)
    print("self.scale", map_image.scale)
    print("self._pixels_per_meter", map_image._pixels_per_meter)


    spawn_points = map.get_spawn_points()
    map_image.draw_spw_pts(spawn_points)

    save_spw_pt(spawn_points, "spw_pt_town5.pt")
    # st_spw_pt_inx = 266
    # ed_spw_pt_inx = 256
    # st_point = spawn_points[st_spw_pt_inx]
    # ed_point = spawn_points[ed_spw_pt_inx]
    # route = map_image.get_route(st_point, ed_point)
    # map_image.draw_route(route)

    #
    st_spw_pt_inx = 266
    ed_spw_pt_inx = 256
    st_point = spawn_points[st_spw_pt_inx]
    ed_point = spawn_points[ed_spw_pt_inx]
    route = map_image.get_route(st_point, ed_point)
    map_image.draw_route(route)

    # st_spw_pt_inx = 264
    # ed_spw_pt_inx = 254
    # st_point = spawn_points[st_spw_pt_inx]
    # ed_point = spawn_points[ed_spw_pt_inx]
    # route = map_image.get_route(st_point, ed_point)
    # map_image.draw_route(route)



    cv2.imwrite("map.png", map_image.map_surface)
    cv2.imwrite("lane.png", map_image.lane_surface)