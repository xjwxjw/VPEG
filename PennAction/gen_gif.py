import numpy as np
import scipy.misc as misc
import scipy.io as io
import imageio
import os

file_list = dict()
cnt = 0
for line in open('./Penn_Action/test_set.txt', 'r'):
    line = line.split(' ')[0].split('/')[1]
    file_list[line] = cnt
    cnt = cnt + 1

# for idx in range(90):
#     img_seq = []
#     for t in range(32):
#         gt = misc.imread('./gif/'+str(idx).zfill(4)+'/real_seq/'+str(t).zfill(6)+'.png')
#         pr = misc.imread('./gif/'+str(idx).zfill(4)+'/pred_seq/'+str(t).zfill(6)+'.png')
#         un = misc.imread('./gif_pred/'+str(idx).zfill(4)+'/pred_seq/'+str(t).zfill(6)+'.png')
#         img = np.concatenate([gt, pr, un], 1)
#         img_seq.append(img)
#     imageio.mimwrite('./test_gif/'+str(idx).zfill(4)+'.gif', img_seq, duration = 0.1, loop = True)

root = './Penn_Action/NN/Test'
cnt = 0
img = None
for name in os.listdir(root):
    data = io.loadmat(os.path.join(root, name))
    img_list = []
    t_idx = name.split('.')[0]
    for t in range(1, 31):
        cur_list = []
        try:
            img = misc.imresize(misc.imread(os.path.join('./Penn_Action/frames', t_idx, str(t).zfill(6)+'.jpg')), (128,128))
        except:
            pass
        cur_list.append(img)
        for idx in data['0'][:2]:
            v_idx, f_idx = idx[0].split('_')
            cur_list.append(misc.imresize(misc.imread(os.path.join('./Penn_Action/', v_idx, str(t+int(f_idx)).zfill(6)+'.jpg')), (128,128)))
        cur_list.append(misc.imresize(misc.imread('./gif_unseen/'+str(file_list[t_idx]).zfill(4)+'/pred_seq/'+str(t-1).zfill(6)+'.png'), (128, 128)))
        cur_list.append(misc.imresize(misc.imread('./gif_pred_unseen/'+str(file_list[t_idx]).zfill(4)+'/pred_seq/'+str(t-1).zfill(6)+'.png'), (128, 128)))
        img_list.append(np.concatenate(cur_list, 1))
    cnt = cnt + 1
    imageio.mimwrite('./demo/'+str(t_idx).zfill(4)+'.gif', img_list, duration = 0.1)      
        

