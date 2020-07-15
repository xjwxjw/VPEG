import numpy as np
import os
import multiprocessing
import scipy.io as io

def func(mp):
    trainset_dict = dict()
    testset_dict = dict()

    for line in open('./Penn_Action/train_set.txt','r'):
        line = line.split('\n')[0]
        dirname, label = line.split(' ')[0], line.split(' ')[1]
        trainset_dict[dirname] = label

    for line in open('./Penn_Action/test_set.txt','r'):
        line = line.split('\n')[0]
        dirname, label = line.split(' ')[0], line.split(' ')[1]
        testset_dict[dirname] = label

    data_dict = dict()
    root = './Penn_Action/pseudo_labels'
    # root = './Penn_Action/labels'
    for filename in os.listdir(root):
        full_path = os.path.join(root, filename)
        # filename = 'frames/'+filename.split('.')[0]
        data = np.load(full_path)
        data_mean = np.mean(data, 1, keepdims=True)
        # data_std = np.std(data, 0, keepdims=True)
        data -= data_mean
        # data = (data - data_mean) #/ (data_std + 1e-8)
        # coor_x = io.loadmat(full_path)['x'].astype(np.float64)
        # coor_x -= (coor_x[:,7:8] + coor_x[:,8:9]) / 2.0
        # coor_y = io.loadmat(full_path)['y'].astype(np.float64)
        # coor_y -= (coor_y[:,7:8] + coor_y[:,8:9]) / 2.0
        # scale = np.max(np.max(coor_y, 1) - np.min(coor_y, 1))
        # data = np.stack([coor_x, coor_y], -1) / scale
        data_dict[filename] = data
        # data_dict[filename] = np.stack([io.loadmat(full_path)['x'], io.loadmat(full_path)['y']], -1)#.copy(    
    query_length = 1
    train_length = 30

    import imageio
    import scipy.misc as misc
    import matplotlib.pyplot as plt

    # cnt = 0
    NNK = 5
    import scipy.io as io

    for i, query_key in enumerate(testset_dict.keys()):
        if not (i >= mp * 3 and i < (mp + 1)* 3):
            continue
        query_label = int(testset_dict[query_key])
        # print(data_dict[query_key.replace('frames/', '')+'.npy'].shape)
        total_length = len(data_dict[query_key.replace('frames/', '')+'.npy'])

        match_dict = dict()
        for query_idx in range(total_length - query_length):

            query_pose = data_dict[query_key.replace('frames/', '')+'.npy'][query_idx:query_idx+query_length]# - data_dict[test_key][:query_length]

            match_vid_dict = dict()

            for train_key in trainset_dict.keys():

                min_fid = -1
                min_diff = 10000000000000000000

                train_label = int(trainset_dict[train_key])
                # if query_key == train_key:
                #     continue
                if (query_label == train_label) and not(query_key == train_key):
                    # continue
                    value_pose = data_dict[train_key.replace('frames/', '')+'.npy']
                    for t in range(value_pose.shape[0]-query_length):
                        tmp_seq = value_pose[t:t+query_length]# - value_pose[t:t+query_length]
                        # tmp_diff = np.mean(np.sqrt((tmp_seq - query_pose) ** 2))
                        tmp_diff = np.mean(np.abs(tmp_seq - query_pose))
                        if min_diff > tmp_diff and ((value_pose.shape[0] - query_length - t - train_length) > 0):
                            min_fid = t
                            min_diff = tmp_diff
                    match_vid_dict[train_key+'_'+str(min_fid)] = np.round(min_diff, 4)
            match_vid_list = sorted(match_vid_dict.items(),key=lambda x:x[1])
            match_dict[str(query_idx)] = match_vid_list[:NNK]
        
        io.savemat('./Penn_Action/NN/Test/'+query_key.replace('frames/', '')+'.mat', match_dict)

if __name__ == "__main__":
    ################## get match matrix ###########
    pool = multiprocessing.Pool(processes=40)
    for i in range(32):
        msg = "hello %d" %(i)
        pool.apply_async(func, (i, ))
    pool.close()
    pool.join()
    print("Sub-process(es) done.")
    

