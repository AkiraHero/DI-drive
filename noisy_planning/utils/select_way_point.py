# import cv2
#
# _world_offset = (-326.0445251464844, -257.8750915527344)
# scale = 1.0
# pixels_per_meter = 10
# map_file = "/home/akira/Project/Model_behaviour/DI-drive/noisy_planning/utils/map_town5.png"
#
# image = cv2.imread(map_file, cv2.IMREAD_COLOR)
# new_scale = 0.2
#
# image = cv2.resize(image, (int(image.shape[1] * new_scale), int(image.shape[0] * new_scale)))
#
#
#
# while True:
#     cv2.imshow("tw5", image)
#     cv2.waitKey(10)
# pass
import tkinter
import tkinter as tk
from PIL import Image, ImageTk


class SpawnPointSelector(tkinter.Frame):
    def __init__(self, master=None):
        super().__init__(master=master)

        map_file = "/home/akira/Project/Model_behaviour/DI-drive/noisy_planning/utils/map.png"
        map_image = Image.open(map_file)
        self.tk_map = ImageTk.PhotoImage(map_image)
        self.scroll_bar_x = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.scroll_bar_y = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.map_canvas = tk.Canvas(self, height=720, width=1280, scrollregion=(0, 0, 6000, 6000))
        self.map_canvas.create_image(0, 0, anchor='nw', image=self.tk_map)

        self.scroll_bar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_bar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.map_canvas.pack()

        self.map_canvas.config(xscrollcommand=self.scroll_bar_x.set, yscrollcommand=self.scroll_bar_y.set)
        self.scroll_bar_y.config(command=self.map_canvas.yview)
        self.scroll_bar_x.config(command=self.map_canvas.xview)

        self.map_canvas.bind("<Button-1>", self.click)
        self.map_canvas.bind_all("<MouseWheel>", self.mouse_scroll)

    def mouse_scroll(self, evt):
        print("scroll", evt)
        # self.map_canvas.yview_scroll(-1*evt.delta,'units')
        self.map_canvas.xview_scroll(-1*evt.delta, 'units')

    def click(self, evt):
        print("clicked!")
        print("x=", self.map_canvas.canvasx(evt.x))
        print("y=", self.map_canvas.canvasy(evt.y))


if __name__ == '__main__':
    root = tk.Tk()
    root.title("SpawnPoint selector")
    root.geometry("1280x720")
    root.focus_set()

    sps = SpawnPointSelector(root, )
    sps.pack()
    root.mainloop()