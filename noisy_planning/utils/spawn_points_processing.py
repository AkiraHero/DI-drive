import numpy as np
import pickle

lane_pts_file1 = "/home/akira/Project/Model_behaviour/DI-drive/noisy_planning/utils/route_1st_lane.txt"
lane_pts_file2 = "/home/akira/Project/Model_behaviour/DI-drive/noisy_planning/utils/route_2nd_lane.txt"
lane_pts_file3 = "/home/akira/Project/Model_behaviour/DI-drive/noisy_planning/utils/route_3th_lane.txt"
lane_files = [lane_pts_file1, lane_pts_file2, lane_pts_file3]

def read_lane_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            pt = [float(i) for i in line.strip('\n').split(",")]
            pts.append(pt)
        return pts

def build_neibor_mat(pt_list):
    len_list = [len(i) for i in pt_list]
    mat_width = sum(len_list)
    neibor_mat = np.zeros([mat_width, mat_width], dtype=np.float)
    pts_all = []
    ids_all = []
    for inx, i in enumerate(pt_list):
        pts_all += i
        ids_all += [inx] * len(i)
    for i in range(mat_width):
        for j in range(i, mat_width):
            pt1 = pts_all[i]
            pt2 = pts_all[j]
            dis = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
            neibor_mat[i, j] = neibor_mat[j, i] = dis

    return pts_all, ids_all, neibor_mat






if __name__ == '__main__':
    lane_pts = []
    lane_ids = []
    for i in lane_files:
        pts = read_lane_file(i)
        lane_pts.append(pts)
    pts_all, ids_all, neibor_mat = build_neibor_mat(lane_pts)
    dat_file = "town5_outter_loop.waypt"
    dat = dict(
        way_pt=pts_all,
        way_pt_id=ids_all,
        neibor_mat=neibor_mat,
    )
    with open(dat_file, 'wb') as f:
        pickle.dump(dat, f)

