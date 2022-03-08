import pickle
import os

def load_pickle(folder):
    files =  [i for i in os.listdir(folder) if ".pickle" in i]
    datas = []
    for i in files:
        with open(os.path.join(folder, i), 'rb') as f:
            datas += pickle.load(f)
    return datas

def cal_mean_rw(d_list):
    rw = 0
    for i in d_list:
        rw += i['reward']
    return rw / len(d_list)

def cal_suc_rate(d_list):
    suc = 0
    for i in d_list:
        if i['success']:
            suc += 1
    return suc / len(d_list)

root_dir = "/home/akira/Project/Model_behaviour/DI-drive/eval"

train_condition = ['withdet', 'nodet', 'nodynamic']
test_condition =  ['withdet', 'nodet', 'detwithdynamic10']

subdirs = os.listdir(root_dir)
for i in subdirs:
    sub_dir = os.path.join(root_dir, i)
    if os.path.isdir(sub_dir):
        episodes = load_pickle(sub_dir)[:500]
        # assert len(episodes) == 500
        suc_rate = cal_suc_rate(episodes)
        mean_reward = cal_mean_rw(episodes)
        print("======={}==========".format(i))
        print("success rate:{}".format(suc_rate))
        print("average rewars:{}".format(mean_reward))
        print("valid episode:{}".format(len(episodes)))

