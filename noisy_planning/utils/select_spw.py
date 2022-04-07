import cv2
import pickle
import copy
# offset = [-52.059906005859375, -52.04996085166931] # town1
offset = (-326.0445251464844, -257.8750915527344) # town5
cnt = 0

# def read_fulltownv():
#     # filename = "/home/juxiaoliang/Project/DI-drive/core/data/benchmark/corl2017/099/full_Town01.txt"
#     filename = "/home/juxiaoliang/Project/DI-drive/core/data/benchmark/corl2017/099/turn_Town01.txt"
#     items = []
#     with open(filename) as f:
#         for line in f.readlines():
#             item = [int(i.strip('\n')) for i in line.split(" ")]
#             items.append(item)
#
#     return items
#
#
def world_to_pixel(location):
    x = 10 * \
        (location[0] - offset[0])
    y = 10 * \
        (location[1] - offset[1])
    return [int(x), int(y)]
#
# def mouse_click(ent, x, y, flags=None, param=None):
#     if ent == cv2.EVENT_LBUTTONDBLCLK:
#         dis_list = []
#         for i in spw_pts:
#             map_loc = world_to_pixel(i)
#             dis_ = ((map_loc[1] - x/mapscale) ** 2 + (map_loc[0] - y/mapscale) ** 2) ** 0.5
#             dis_list.append(dis_)
#         inx = dis_list.index(min(dis_list))
#         with open("left_mid_turn.selct", 'a') as f:
#             s = "{},{:.2f},{:.2f},{:.2f}，dis={:.2f}".format(inx, spw_pts[inx][0], spw_pts[inx][1], spw_pts[inx][2], min(dis_list))
#             print(s)
#             f.write(s + '\n')
#             param['cnt'] += 1
#             if param['cnt'] % 2 == 0:
#                 print('---')
#                 f.write('---' + '\n')


map_img = cv2.imread("./town5_spw.png")
with open("./spw_pts_t5.pickle", 'rb') as f:
    spw_pts = pickle.load(f)
# redraw points to ensure
cv2.flip(map_img, 0, map_img)
mapscale = 1.0
map_img = cv2.resize(map_img, (int(map_img.shape[0] * mapscale), int(map_img.shape[1] * mapscale)))

for inx, i in enumerate(spw_pts):
    wx = i[0]
    wy = i[1]
    [py, px] = world_to_pixel([wx, wy])
    px = int(px * mapscale)
    py = int(py * mapscale)
    py = map_img.shape[0] - py
    cv2.circle(map_img, [px, py], 10, (0, 255, 0), 20)

for inx, i in enumerate(spw_pts):
    wx = i[0]
    wy = i[1]
    [py, px] = world_to_pixel([wx, wy])
    px = int(px * mapscale)
    py = int(py * mapscale)
    py = map_img.shape[0] - py
    cv2.putText(map_img, str(inx), [px - 12, py], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 26, 253), 2)

cv2.imwrite("spawn_point_town5.png", map_img)
# cv2.imshow("dd", map_img)
# cv2.waitKey(-1)



mapscale = 0.21


# routs = read_fulltownv()
# for i in routs:
#     i = routs[18]
#     map_img_draw = copy.deepcopy(map_img)
#     wx = spw_pts[i[0]][0]
#     wy = spw_pts[i[0]][1]
#     [py, px] = world_to_pixel([wx, wy])
#     cv2.circle(map_img_draw, [px, py], 30, (0, 0, 255), 30)
#
#     wx = spw_pts[i[1]][0]
#     wy = spw_pts[i[1]][1]
#     [py, px] = world_to_pixel([wx, wy])
#     cv2.circle(map_img_draw, [px, py], 30, (0, 255, 255), 30)
#
#     map_img_show = cv2.resize(map_img_draw, (int(map_img_draw.shape[0] * mapscale), int(map_img_draw.shape[1] * mapscale)))
#
    # cv2.imshow("map", map_img_show)
    # cv2.waitKey(-1)

# cv2.setMouseCallback("map", mouse_click, param={'cnt':cnt})
# cv2.waitKey(-1)