import os
import io
from scipy.misc import imresize 
import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread 
import csv

class RobotPushMatch(object):
    
    """Data Handler that loads robot pushing data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64):
        self.root_dir = data_root 
        self.train = train
        self.seq_match = []
        if train:
            self.data_dir = self.root_dir+'/train'
            for line in open('train_match.txt'):
                idx_array = np.array([int(idx) for idx in line.split('\n')[0].split(' ')[1:-1]])
                self.seq_match.append(idx_array)
        else:
            self.data_dir = self.root_dir+'/test'
            for line in open('test_match.txt'):
                idx_array = np.array([int(idx) for idx in line.split('\n')[0].split(' ')[1:-1]])
                self.seq_match.append(idx_array)
        
        self.dirs_test = []
        self.dirs_train = []
        for i in range(43264):
            self.dirs_train.append('%s/%s' % (self.data_dir.replace('test', 'train'), str(i).zfill(5)))
        for i in range(256):
            self.dirs_test.append('%s/%s' % (self.data_dir.replace('train', 'test'), str(i).zfill(3)))
        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False 
        self.d = 0
        self.cur_cnt = 0

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return 10000

    def get_seq(self):
        if self.train:
            idx = np.random.randint(len(self.dirs_train))
            d = self.dirs_train[idx]
        else:
            idx = np.random.randint(len(self.dirs_test))
            d = self.dirs_test[idx]
        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/aux1_frames/frame_%03d.bmp' % (d, i)
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
        image_seq = np.concatenate(image_seq, axis=0)

        image_seq_match_list = []
        for m in range(0,5):
            image_seq_match = []
            d = self.dirs_train[self.seq_match[idx][m]]
            for i in range(self.seq_len):
                fname = '%s/aux1_frames/frame_%03d.bmp' % (d, i)
                im = imread(fname).reshape(1, 64, 64, 3)
                image_seq_match.append(im/255.)
            image_seq_match = np.concatenate(image_seq_match, axis=0)
            image_seq_match_list.append(image_seq_match)
        return image_seq, image_seq_match_list


    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()

class RobotPush(object):
    
    """Data Handler that loads robot pushing data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64):
        self.root_dir = data_root 
        if train:
            self.data_dir = self.root_dir+'/train'
        else:
            self.data_dir = self.root_dir+'/test'
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            self.dirs.append('%s/%s' % (self.data_dir, d1))
        # for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
        #     self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False # multi threaded loading
        self.d = 0

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return 10000

    def get_seq(self):
        d = self.dirs[np.random.randint(len(self.dirs))]
        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/aux1_frames/frame_%03d.bmp' % (d, i)
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
        image_seq = np.concatenate(image_seq, axis=0)
        return image_seq


    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()


