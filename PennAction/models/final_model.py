import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base_model import BaseModel
from . import networks
from utils import model as model_utils

N_FUTURE_FRAMES = 32
IMAGE_SIZE = 128


class FinalModel(BaseModel):
    """
    Models for final evaluation.
    This model is not trainable.
    """

    name = 'final'

    def __init__(self, config):
        super(FinalModel, self).__init__(False)

        # configuration variables
        model_config = config['model']
        paths_config = config['paths']

        self.log_dir = paths_config['log_dir']

        # model config variables
        self.n_points = model_config['n_pts']
        self.cell_info = model_config['cell_info']
        self.vae_dim = model_config['vae_dim']
        self.sth_pro = model_config['sth_pro']

        self.colors = model_utils.get_n_colors(model_config['n_pts'], pastel_factor=0.0)

        # inputs
        self.input_im = None
        self.input_im_seq = None
        self.input_action_code = None
        self.input_real_seq = None

        # outputs
        self.output_ops = None
        pass

    def build(self, inputs):
        # input setup
        self.input_im = inputs['image']
        self.input_im_seq = inputs['real_im_seq']
        self.input_action_code = inputs['action_code']
        self.input_real_seq = inputs['real_seq']
        self.input_ref_seq = tf.reshape( inputs['ref_seq'], [-1, 5, N_FUTURE_FRAMES, self.n_points * 2])
        self.mean_seq = inputs['mean_seq']

        # forward pass
        im = self.input_im
        tiled_im = tf.expand_dims(im, 1)
        tiled_im = tf.tile(tiled_im, [1, 32, 1, 1, 1])
        tiled_im = tf.reshape(tiled_im, [-1, 128, 128, 3])

        embeddings = networks.image_encoder(im, self.is_training)
        im_emb_size = embeddings[-2].shape.as_list()[-3:]
        embeddings = tf.expand_dims(embeddings[-2], 1)
        embeddings = tf.tile(embeddings, [1, 32, 1, 1, 1])
        embeddings = tf.reshape(embeddings, [-1] + im_emb_size)

        first_pt = inputs['keypoints']# networks.pose_encoder(im, self.n_points, self.is_training)
        first_pt = tf.reshape(first_pt, [-1, self.n_points * 2])
        # print(first_pt.get_shape())
        # print(self.input_im_seq.get_shape())
        # pred_seq = None
        # if False:
        #     pred_list = []
        #     for t in range(self.input_im_seq.get_shape()[1]):
        #         cur_pt = networks.pose_encoder(self.input_im_seq[:,t], self.n_points, self.is_training)
        #         cur_pt = tf.expand_dims(tf.reshape(cur_pt, [-1, self.n_points, 2]), 1)
        #         pred_list.append(cur_pt)
        #     pred_seq = tf.concat(pred_list, 1)
        # else:

        #     gt_list = []
        #     for t in range(self.input_im_seq.get_shape()[1]):
        #         cur_pt = networks.pose_encoder(self.input_im_seq[:,t], self.n_points, self.is_training)
        #         cur_pt = tf.expand_dims(tf.reshape(cur_pt, [-1, self.n_points, 2]), 1)
        #         gt_list.append(cur_pt)
        #     gt_seq = tf.concat(gt_list, 1)

        #     z = tf.random_normal([tf.shape(first_pt)[0]] + [self.vae_dim], 0, 1, dtype=tf.float32)
        #     pred_seq = networks.vae_decoder(z, first_pt,
        #                                     self.input_action_code,
        #                                     self.cell_info,
        #                                     self.vae_dim,
        #                                     self.n_points)
        #     pred_seq = (0.5 * tf.reshape(pred_seq, [-1, 32, self.n_points, 2]) + 0.5 * gt_seq)
        if not self.sth_pro:
            # mu, stddev = networks.vae_encoder(tf.reshape(self.input_real_seq, [-1, N_FUTURE_FRAMES, self.n_points * 2]),
            #                                     first_pt,
            #                                     self.input_action_code,
            #                                     self.cell_info,
            #                                     self.vae_dim,
            #                                     self.input_ref_seq)
            z = tf.random_normal([tf.shape(first_pt)[0]] + [self.vae_dim], 0, 1, dtype=tf.float32)
            pred_seq = networks.vae_decoder(z,
                                            first_pt,
                                            self.input_action_code,
                                            self.cell_info,
                                            self.vae_dim,
                                            self.n_points)
        else:
            mu, stddev = networks.vae_encoder(tf.reshape(self.input_real_seq, [-1, N_FUTURE_FRAMES, self.n_points * 2]),
                                              first_pt,
                                              self.input_action_code,
                                              self.cell_info,
                                              self.vae_dim,
                                              self.input_ref_seq)
            # mu, stddev = tf.nn.moments(self.input_ref_seq, axes = 1, keep_dims=False)
            for _ in range(1):
                z = mu + stddev * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
                pred_seq = networks.vae_decoder(z,
                                            first_pt,
                                            self.input_action_code,
                                            self.cell_info,
                                            self.vae_dim,
                                            self.n_points,
                                            self.input_ref_seq)
                # pred_seq = tf.reshape(tf.reshape(pred_seq, [-1, N_FUTURE_FRAMES, self.n_points, 2]) + self.mean_seq, [-1, N_FUTURE_FRAMES, self.n_points*2])
                # pred_seq.append(tmp)

        current_pt_map = model_utils.get_gaussian_maps(tf.reshape(first_pt,
                                                                  [-1,
                                                                   self.n_points,
                                                                   2]),
                                                       [32, 32])
        emb_size = current_pt_map.shape.as_list()[-3:]
        current_pt_map = tf.expand_dims(current_pt_map, 1)
        current_pt_map = tf.tile(current_pt_map, [1, 32, 1, 1, 1])
        current_pt_map = tf.reshape(current_pt_map, [-1] + emb_size)
        pred_pt_map = model_utils.get_gaussian_maps(tf.reshape(pred_seq,
                                                               [-1,
                                                                self.n_points,
                                                                2]),
                                                    [32, 32])

        joint_embedding = tf.concat([embeddings, current_pt_map, pred_pt_map], axis=-1)
        crude_output, mask = networks.translator(joint_embedding, self.is_training)
        final_output = tiled_im * mask + crude_output * (1 - mask)
        
        crude_output = tf.clip_by_value(crude_output, -1, 1)
        final_output = tf.clip_by_value(final_output, -1, 1)

        # visualization outputs
        current_keypoints_map = model_utils.get_gaussian_maps(tf.reshape(first_pt, [-1, self.n_points, 2]),
                                                              [128, 128])
        future_keypoints_map = model_utils.get_gaussian_maps(tf.reshape(pred_seq, [-1, self.n_points, 2]),
                                                             [128, 128])
        current_points = tf.reshape(model_utils.colorize_point_maps(current_keypoints_map, self.colors),
                                    [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
        future_points = tf.reshape(model_utils.colorize_point_maps(future_keypoints_map, self.colors),
                                   [-1, N_FUTURE_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3])

        # final output tensors
        if self.sth_pro:
            self.output_ops = {
                'first_pt': first_pt,
                'real_im_seq': self.input_im_seq,
                'real_seq': tf.reshape(self.input_real_seq, [-1, N_FUTURE_FRAMES, self.n_points * 2]),
                'im': self.input_im,
                'pred_im_seq': tf.reshape(final_output, [-1, N_FUTURE_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3]),
                'mask': tf.reshape(mask, [-1, N_FUTURE_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 1]),
                'pred_im_crude': tf.reshape(crude_output, [-1, N_FUTURE_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3]),
                'current_points': current_points,
                'future_points': future_points,
                'fut_pt_raw': pred_seq,
                'ref_seq': self.input_ref_seq,
                'z': stddev
            }
        else:
            self.output_ops = {
            'real_im_seq': self.input_im_seq,
            'real_seq': tf.reshape(self.input_real_seq, [-1, N_FUTURE_FRAMES, self.n_points * 2]),
            'im': self.input_im,
            'pred_im_seq': tf.reshape(final_output, [-1, N_FUTURE_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3]),
            'mask': tf.reshape(mask, [-1, N_FUTURE_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 1]),
            'pred_im_crude': tf.reshape(crude_output, [-1, N_FUTURE_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3]),
            'current_points': current_points,
            'future_points': future_points,
            'fut_pt_raw': pred_seq
        }
        pass

    def run(self, sess, feed_dict):
        return sess.run(self.output_ops, feed_dict=feed_dict)

    def train_step(self,
                   sess,
                   feed_dict,
                   step,
                   batch_size,
                   should_write_log=False, should_write_summary=False):
        """
        This model is not trainable
        """
        raise NotImplementedError

    def test_step(self, sess, feed_dict, step, test_idx, batch_size):
        """
        This model has no test step
        """
        raise NotImplementedError

    def collect_test_results(self, results, step):
        """
        This model has no test step
        """
        raise NotImplementedError
