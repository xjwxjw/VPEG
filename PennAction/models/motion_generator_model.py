import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base_model import BaseModel
from . import networks
from utils import model as model_utils

N_FUTURE_FRAMES = 32


class MotionGeneratorModel(BaseModel):
    # name will be used for log management (defined in BaseModel class)
    name = 'motion_generator'

    def __init__(self, config, global_step=None, is_training=True):
        super(MotionGeneratorModel, self).__init__(is_training)

        # configuration variables
        train_config = config['training']
        model_config = config['model']
        paths_config = config['paths']

        if self.is_training:
            self.lr = train_config['lr']
        else:
            self.lr = None
        self.batch_size = train_config['batch_size']
        self.log_dir = paths_config['log_dir']

        # model config variables
        self.n_points = model_config['n_pts']
        self.cell_info = model_config['cell_info']
        self.vae_dim = model_config['vae_dim']
        self.sth_pro = model_config['sth_pro']

        self.colors = model_utils.get_n_colors(model_config['n_pts'], pastel_factor=0.0)

        # inputs
        self.global_step = global_step
        self.input_im = None
        self.init_keypoints = None
        self.input_real_seq = None
        self.input_ref_seq = None
        self.input_action_code = None

        # outputs
        self.current_lr = None
        self.pred_seq = None

        # losses
        self.loss_D_real = None
        self.loss_D_fake = None
        self.loss_D = None
        self.loss_G_recon = None
        self.loss_G_kl = None
        self.loss_G_adv = None
        self.loss_reg = None
        self.loss_G = None

        # summaries
        self.summary_lr = None
        self.summary_image = None
        self.summary_d_loss = None
        self.summary_g_loss = None
        pass

        self.cnt_step = 0

    def build(self, inputs):
        # input setup
        self.input_im = inputs['image']
        self.init_keypoints = inputs['keypoints']
        self.input_real_seq = inputs['real_seq']
        self.input_ref_seq = inputs['ref_seq']
        self.input_action_code = inputs['action_code']

        self._define_forward_pass()
        if self.is_training:
            self._compute_loss()
            self._define_summary()
        pass

    def train_step(self,
                   sess,
                   feed_dict,
                   step,
                   batch_size,
                   should_write_log=False, should_write_summary=False):
        D_ops = [self.loss_D, self.train_op_D]
        G_ops = [self.loss_G, self.train_op_G]

        if should_write_summary:
            D_ops.extend([self.summary_d_loss])
            G_ops.extend([self.summary_g_loss, self.summary_image, self.summary_lr])

        # G_ops.extend([self.ref_seq, self.real_seq])
        start_time = time.time()
        D_values = sess.run(D_ops, feed_dict=feed_dict)
        G_values = sess.run(G_ops, feed_dict=feed_dict)
        duration = time.time() - start_time
        
        if should_write_log:
            examples_per_sec = batch_size / float(duration)
            loss_value_D = D_values[0]
            loss_value_G = G_values[0]
            log_format = '%s: step %d, loss_D = %.4f, loss_G = %.4f (%.1f examples/sec) %.3f sec/batch'
            tf.logging.info(log_format % (datetime.now(),
                                          step,
                                          loss_value_D,
                                          loss_value_G,
                                          examples_per_sec,
                                          duration))
        if should_write_summary:
            summary_loss_D = D_values[2]
            summary_loss_G = G_values[2]
            summary_image = G_values[3]
            summary_lr = G_values[4]
            self.train_writer.add_summary(summary_loss_D, step)
            self.train_writer.add_summary(summary_loss_G, step)
            self.train_writer.add_summary(summary_image, step)
            self.train_writer.add_summary(summary_lr, step)
        pass

    def test_step(self, sess, feed_dict, step, test_idx, batch_size):
        ops = [self.loss_D, self.loss_G]
        should_write_summary = test_idx == 0

        if should_write_summary:
            ops.extend([self.summary_d_loss, self.summary_g_loss, self.summary_image])

        start_time = time.time()
        values = sess.run(ops, feed_dict=feed_dict)
        duration = time.time() - start_time

        if should_write_summary:
            summary_loss_D = values[2]
            summary_loss_G = values[3]
            summary_image = values[4]
            self.test_writer.add_summary(summary_loss_D, step)
            self.test_writer.add_summary(summary_loss_G, step)
            self.test_writer.add_summary(summary_image, step)

        loss_value_D = values[0]
        loss_value_G = values[1]

        return loss_value_D, loss_value_G, duration, batch_size

    def collect_test_results(self, results, step):
        average_loss_D = sum([x[0] for x in results]) / len(results)
        average_loss_G = sum([x[1] for x in results]) / len(results)
        total_duration = sum([x[2] for x in results])
        average_duration = total_duration / len(results)
        num_examples = sum([x[3] for x in results])
        examples_per_sec = num_examples / total_duration

        log_format = 'test: %s: step %d, loss_D = %.4f, loss_G = %.4f (%.1f examples/sec) %.3f sec/batch'
        tf.logging.info(log_format % (datetime.now(),
                                      step,
                                      average_loss_D,
                                      average_loss_G,
                                      examples_per_sec,
                                      average_duration))
        pass

    def _define_forward_pass(self):
        init_keypoints = self.init_keypoints
        real_seq = self.input_real_seq
        action_code = self.input_action_code
        ref_seq = self.input_ref_seq
        pred_seq = []
        sample_z = []

        first_pt = tf.reshape(init_keypoints, [-1, self.n_points * 2])
        ref_seq = tf.reshape(ref_seq, [-1, 5, N_FUTURE_FRAMES, self.n_points * 2])
        if self.is_training:
            if self.sth_pro:
                # mu, stddev = tf.nn.moments(ref_seq, axes = 1, keep_dims=False)
                mu, stddev = networks.vae_encoder(tf.reshape(real_seq, [-1, N_FUTURE_FRAMES, self.n_points * 2]),
                                              first_pt,
                                              action_code,
                                              self.cell_info,
                                              self.vae_dim,
                                              ref_seq)
                ## Our work: at each training iteration, obtain 5 stochastic predictions. ##
                for _ in range(5):
                    z = mu + stddev * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
                    tmp = networks.vae_decoder(z,
                                                first_pt,
                                                action_code,
                                                self.cell_info,
                                                self.vae_dim,
                                                self.n_points,
                                                ref_seq)
                    pred_seq.append(tmp)
                    sample_z.append(z)
            else:
                mu, stddev = networks.vae_encoder(tf.reshape(real_seq, [-1, N_FUTURE_FRAMES, self.n_points * 2]),
                                              first_pt,
                                              action_code,
                                              self.cell_info,
                                              self.vae_dim,
                                              ref_seq)
                z = mu + stddev * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
                pred_seq = networks.vae_decoder(z,
                                                first_pt,
                                                action_code,
                                                self.cell_info,
                                                self.vae_dim,
                                                self.n_points,
                                                None)
        else:
            if self.sth_pro:
                mu, stddev = tf.nn.moments(ref_seq, axes = 1, keep_dims=False)
                for _ in range(1):
                    z = mu + stddev * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
                    tmp = networks.vae_decoder(z,
                                                first_pt,
                                                action_code,
                                                self.cell_info,
                                                self.vae_dim,
                                                self.n_points,
                                                ref_seq)
                    pred_seq.append(tmp)
            else:
                z = mu + stddev * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
                pred_seq = networks.vae_decoder(z,
                                                first_pt,
                                                action_code,
                                                self.cell_info,
                                                self.vae_dim,
                                                self.n_points,
                                                None)

        # outputs
        self.pred_seq = pred_seq
        self.ref_seq = ref_seq
        self.sample_z = sample_z
        if self.is_training:
            self.mu = mu
            self.stddev = stddev
        pass

    def _compute_loss(self):
        real_seq = tf.reshape(self.input_real_seq, [-1, N_FUTURE_FRAMES, self.n_points * 2])
        self.real_seq = real_seq
        if self.sth_pro:
            self._compute_loss_D(self.pred_seq[0], real_seq)
            self._compute_loss_G(self.pred_seq, real_seq, self.mu, self.stddev, self.ref_seq)
        else:
            self._compute_loss_D(self.pred_seq, real_seq)
            self._compute_loss_G(self.pred_seq, real_seq, self.mu, self.stddev)

        # optimization
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'discr' not in var.name]
        D_vars = [var for var in t_vars if 'discr' in var.name]
        lr = tf.train.exponential_decay(self.lr['start_val'], self.global_step, self.lr['step'],
                                        self.lr['decay'])

        self.current_lr = lr
        self.train_op_D = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_D,
                                                                                      var_list=D_vars)
        self.train_op_G = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_G,
                                                                                      var_list=G_vars,
                                                                                      global_step=self.global_step)
        pass

    def _define_summary(self):
        # D loss summary
        summary_d_real = tf.summary.scalar('D_real', self.loss_D_real)
        summary_d_fake = tf.summary.scalar('D_fake', self.loss_D_fake)
        summary_d_loss = tf.summary.merge([summary_d_real, summary_d_fake])

        # G loss summary
        summary_g_kl = tf.summary.scalar('kl_loss', self.loss_G_kl)
        summary_g_recon = tf.summary.scalar('recon_loss', self.loss_G_recon)
        summary_g_adv = tf.summary.scalar('G_adv_loss', self.loss_G_adv)
        # summary_g_reg = tf.summary.scalar('reg_loss', self.loss_reg)
        summary_g_loss = tf.summary.merge([summary_g_recon, summary_g_kl, summary_g_adv])

        # final summaries to write
        self.summary_lr = tf.summary.scalar('lr', self.current_lr)
        self.summary_image = self._get_image_visualization_summary()
        self.summary_d_loss = summary_d_loss
        self.summary_g_loss = summary_g_loss
        pass

    def _get_image_visualization_summary(self):
        pred_seq = self.pred_seq[0]
        real_seq = self.input_real_seq
        init_keypoints = self.init_keypoints

        # convert pred_seq to images
        pred_seq_img = []
        for i in range(N_FUTURE_FRAMES):
            gauss_map = model_utils.get_gaussian_maps(tf.reshape(pred_seq[:, i, ::], [-1, self.n_points, 2]),
                                                      [64, 64])
            pred_seq_img.append(model_utils.colorize_point_maps(gauss_map, self.colors))
            pass
        pred_seq_img = tf.concat(pred_seq_img, axis=2)

        # convert real_seq to images
        real_seq_img = []
        for i in range(N_FUTURE_FRAMES):
            gauss_map = model_utils.get_gaussian_maps(real_seq[:, i, ::], [64, 64])
            real_seq_img.append(model_utils.colorize_point_maps(gauss_map, self.colors))
            pass
        real_seq_img = tf.concat(real_seq_img, axis=2)

        first_pt_map = model_utils.get_gaussian_maps(tf.reshape(init_keypoints, [-1, self.n_points, 2]), [128, 128])

        # image summary
        summary_im = tf.summary.image('im', (self.input_im + 1) / 2.0 * 255.0, max_outputs=2)
        summary_first_pt = tf.summary.image('first_pt',
                                            model_utils.colorize_point_maps(first_pt_map, self.colors),
                                            max_outputs=2)
        summary_pred_p_seq = tf.summary.image('predicted_pose_sequence',
                                              model_utils.colorize_point_maps(pred_seq_img, self.colors),
                                              max_outputs=2)
        summary_real_p_seq = tf.summary.image('real_pose_sequence',
                                              model_utils.colorize_point_maps(real_seq_img, self.colors),
                                              max_outputs=2)

        return tf.summary.merge([summary_im,
                                 summary_first_pt,
                                 summary_pred_p_seq,
                                 summary_real_p_seq])

    def _compute_loss_D(self, pred_seq, real_seq):
        real_ = networks.seq_discr(real_seq)
        fake_ = networks.seq_discr(pred_seq)
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_), logits=real_)
        )
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_), logits=fake_)
        )
        loss = real_loss + fake_loss

        self.loss_D_real = real_loss
        self.loss_D_fake = fake_loss
        self.loss_D = loss
        pass

    def _compute_loss_G(self, pred_seq, real_seq, mu, stddev, ref_seq = None):
        if self.sth_pro:
            ## Our work: match with the expectation, select the best predicted one to train the model ##
            mse_array = [tf.reduce_mean(tf.abs(cur_pred - real_seq), axis = [-2,-1], keepdims = True) for cur_pred in pred_seq]
            mse_tensor = tf.concat(axis = -1, values = mse_array) 
            mask = tf.one_hot(indices = tf.argmin(mse_tensor, axis = -1), depth = 5, axis = -1, on_value = 1.0, off_value = 0.0)
            mask = tf.split(mask, 5, axis = -1)
            mask = tf.stop_gradient(mask)
            final_pred = tf.add_n([pred_seq[k] * tf.tile(mask[k], [1, N_FUTURE_FRAMES, self.n_points * 2]) for k in range(5)])
            l = 1000 * tf.abs(final_pred - real_seq) 
        else:
            l = 1000 * tf.abs(pred_seq - real_seq)
        pred_seq_loss = tf.reduce_mean(l)
        if self.sth_pro:
            ## Our work: match with the variance, the variance of best predicted one shold match with one of the reference sequences ##
            pred_var = tf.tile(tf.expand_dims(pred_seq[0][:,1:] - pred_seq[0][:,:(N_FUTURE_FRAMES-1)], 1), [1, 5, 1, 1])
            ref_var = ref_seq[:,:,1:] - ref_seq[:,:,:(N_FUTURE_FRAMES-1)]
            mse_var = tf.reduce_mean(tf.abs(pred_var - ref_var), [-2, -1], keepdims = True)
            mask = tf.one_hot(indices = tf.argmin(mse_var, axis = 1), depth = 5, axis = 1, on_value = 1.0, off_value = 0.0)
            mask = tf.split(mask, 5, axis = 1)
            mask = tf.stop_gradient(mask)
            ref_seq_final = tf.add_n([ref_seq[:,k] * tf.tile(mask[k,:,0], [1, N_FUTURE_FRAMES, self.n_points * 2]) for k in range(5)])
            var_loss = 10 * tf.abs(pred_var[:,0] - (ref_seq_final[:,1:] - ref_seq_final[:,:(N_FUTURE_FRAMES-1)]))

            _, pred_std = tf.nn.moments(pred_var, [1])
            _, red_std = tf.nn.moments(ref_var, [1])
            ## Our work: match with the variance, the variance of 5 randomly predicted results shold match with the reference ##
            kl_l = 0.5 * tf.reduce_mean(tf.abs(pred_std - red_std))
        else:
            kl_l = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(stddev) - tf.log(1e-8 + tf.square(stddev)) - 1, 1)
        kl_loss = tf.reduce_mean(kl_l) + tf.reduce_mean(var_loss)
        fake_ = networks.seq_discr(pred_seq[0])
        adv_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_), logits=fake_)
        )

        self.loss_G_recon = pred_seq_loss
        self.loss_G_kl = kl_loss
        self.loss_G_adv = adv_loss
        self.loss_G = kl_loss + pred_seq_loss + adv_loss
        pass
