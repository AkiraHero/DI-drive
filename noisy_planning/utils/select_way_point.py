import sys
sys.path.append("/home/akira/carla-0.9.11-py3.7-linux-x86_64.egg")
import carla
import tkinter
import tkinter as tk
from PIL import Image, ImageTk

from core.utils.simulator_utils.carla_agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from core.utils.simulator_utils.carla_agents.navigation.global_route_planner import GlobalRoutePlanner


class SpawnPointSelector(tkinter.Frame):
    def __init__(self, master=None):
        super().__init__(master=master)
        map_file = "/home/akira/Project/Model_behaviour/DI-drive/noisy_planning/utils/map.png"
        map_image = Image.open(map_file)
        xodr_file = "/home/akira/Project/Model_behaviour/DI-drive/noisy_planning/utils/town5.xodr"
        xodr = open(xodr_file, 'r').read()
        map_name = "town5"
        self._carla_map = carla.Map(map_name, xodr)
        self._map_resolution = 1.0
        self._grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(self._carla_map, self._map_resolution))
        self._grp.setup()
        self._map_scale = 1.0
        self._map_offset = (-326.0445251464844, -257.8750915527344)
        self._pixels_per_meter = 10
        self.tk_map = ImageTk.PhotoImage(map_image)
        self.scroll_bar_x = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.scroll_bar_y = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.map_canvas = tk.Canvas(self, height=720, width=1280, scrollregion=(0, 0, 6000, 6000))
        self.map_canvas.create_image(0, 0, anchor='nw', image=self.tk_map)

        self.scroll_bar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_bar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.map_canvas.config(xscrollcommand=self.scroll_bar_x.set, yscrollcommand=self.scroll_bar_y.set)
        self.scroll_bar_y.config(command=self.map_canvas.yview)
        self.scroll_bar_x.config(command=self.map_canvas.xview)

        self.map_canvas.bind("<Button-1>", self.click)
        self.map_canvas.bind_all("<MouseWheel>", self.mouse_scroll)

        self._finish_button = tk.Button(self, text="finish", command=self.on_click_finish_button)
        self._finish_button.pack()
        self.map_canvas.pack()

        self._start_point = None
        self._new_point = None
        self._route_cnt = 0
        self._cur_route = []

    def on_click_finish_button(self):
        self._start_point = None
        self._new_point = None
        self.map_canvas.create_image(0, 0, anchor='nw', image=self.tk_map)
        self.write2file(self._cur_route)
        self._cur_route.clear()

    def map2world(self, x, y):
        factor = self._map_scale * self._pixels_per_meter
        gx = x / factor + self._map_offset[0]
        gy = y / factor + self._map_offset[1]
        return gx, gy

    def world2map(self, x, y):
        factor = self._map_scale * self._pixels_per_meter
        px = factor * (x - self._map_offset[0])
        py = factor * (y - self._map_offset[1])
        return px, py

    def mouse_scroll(self, evt):
        print("scroll", evt)
        # self.map_canvas.yview_scroll(-1*evt.delta,'units')
        self.map_canvas.xview_scroll(-1*evt.delta, 'units')

    def get_seg_route(self, st, ed):
        st = carla.Location(*st)
        ed = carla.Location(*ed)
        start_waypoint = self._carla_map.get_waypoint(st)
        end_waypoint = self._carla_map.get_waypoint(ed)
        new_route = self._grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        return new_route

    def draw_point(self, mapx, mapy, radius=10):
        x1, y1 = mapx - radius, mapy - radius
        x2, y2 = mapx + radius, mapy + radius
        self.map_canvas.create_oval(x1, y1, x2, y2, fill="#476042")

    def draw_route(self, route):
        pt_list = []
        for i in route:
            location = (i[0].transform.location.x, i[0].transform.location.y)
            pt_list.append(location)
        # trans to pixel
        map_pt_list = []
        for i in pt_list:
            pixel_loc = self.world2map(i[0], i[1])
            map_pt_list.append(pixel_loc)
        for i in range(len(map_pt_list) - 1):
            st = map_pt_list[i]
            ed = map_pt_list[i + 1]
            self.map_canvas.create_line(st[0], st[1], ed[0], ed[1], fill="green", width=5)
        pass

    def write2file(self, route):
        file_name = "route_{}.txt".format(self._route_cnt)
        with open(file_name, 'w') as f:
            for i in route:
                # x,y,z,yaw
                x = i[0].transform.location.x
                y = i[0].transform.location.y
                z = i[0].transform.location.z
                yaw = i[0].transform.rotation.yaw
                f.write("{:.3f},{:.3f},{:.3f},{:.3f}\n".format(x, y, z ,yaw))
        self._route_cnt += 1


    def click(self, evt):
        print("clicked!")
        print("x=", self.map_canvas.canvasx(evt.x))
        print("y=", self.map_canvas.canvasy(evt.y))
        map_x = self.map_canvas.canvasx(evt.x)
        map_y = self.map_canvas.canvasy(evt.y)
        self.draw_point(map_x, map_y)
        if self._start_point is None:
            self._start_point = self.map2world(map_x, map_y)
        else:
            self._new_point = self.map2world(map_x, map_y)
            route = self.get_seg_route(self._start_point, self._new_point)
            self.draw_route(route)
            self._cur_route += route
            self._start_point = self._new_point


if __name__ == '__main__':
    root = tk.Tk()
    root.title("SpawnPoint selector")
    root.geometry("1280x720")
    root.focus_set()

    sps = SpawnPointSelector(root, )
    sps.pack()
    root.mainloop()