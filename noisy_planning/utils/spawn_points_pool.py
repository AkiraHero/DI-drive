import pickle
import numpy as np


class SpawnPointsPool(object):
    def __init__(self, dat):
        with open(dat, 'rb') as f:
            dat = pickle.load(f)
            self._neigb_mat = dat['neibor_mat']
            self._way_point_list = dat['way_pt']
        self._total_point = len(self._way_point_list)
        self._candidate_indices = {i: None for i in range(self._total_point)}

    def reset(self):
        self._candidate_indices = np.ones([self._total_point])

    def get_indices_in_radius(self, pt_inx, radius):
        dis_list = self._neigb_mat[pt_inx]
        indices, = (dis_list < radius).nonzero()
        return indices

    def select_points(self, num, radiu=5.0, reset=True):
        selected_indices = []
        while len(selected_indices) < num:
            if len(self._candidate_indices) == 0:
                break
            valid_candidate = [i for i in self._candidate_indices.keys()]
            inx = np.random.choice(valid_candidate)
            selected_indices.append(inx)
            invalid_indices = self.get_indices_in_radius(inx, radiu)
            for i in invalid_indices:
                if i in self._candidate_indices.keys():
                    self._candidate_indices.pop(i)
        if reset:
            self.reset()
        return [self._way_point_list[i] for i in selected_indices]

    def __len__(self):
        return self._total_point


if __name__ == '__main__':
    import time
    dat_file = "town5_outter_loop.waypt"
    st_time = time.time()
    s_pool = SpawnPointsPool(dat_file)
    num = 1300
    wpts = s_pool.select_points(num, radiu=5)
    print("select:{}/{}".format(len(wpts), num))
    ed_time = time.time()
    print("time cost:", ed_time - st_time)
