# train_layer_1.py

import tensorflow as tf
from scipy import misc
import os
import json
import fnmatch
import shutil
import numpy as np
import math

from im_modules import utils


class SP_Layer_1():
    def __init__(self, train_dir, test_dir, checkout_dir, batch_size=10):
        self.train_dir_g = os.path.join(train_dir, 'g')
        self.train_dir_ng = os.path.join(train_dir, 'ng')
        self.test_dir_g = os.path.join(test_dir, 'g')
        self.test_dir_ng = os.path.join(test_dir, 'ng')

        self.checkout_dir = checkout_dir
        self.batch_size = 50

    def run_train(self):
        print('train started')

        train_imgs, train_vecs, test_imgs, test_vecs = self.load_data()
        if train_imgs is None or test_imgs is None:
            return

        print('init session')
        sess = tf.Session()

        img_res = 144
        x = tf.placeholder(tf.float32, shape=[None, img_res, img_res, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 4])

        print('-- convolutions --')

        # CL0 (144 -> 72)
        h_conv0 = self.conv(x, 3, 64, strides=[1, 2, 2, 1])
        print('h_conv0: ', h_conv0)

        # RB1 (72 -> 72)
        h_rb1 = h_conv0
        for resi in range(3):
            h_rb1 = self.resblock(h_rb1, 64)
            print('h_rb1: ', h_rb1)

        # CL1 (72 -> 36)
        h_conv1 = self.conv(h_rb1, 64, 128, strides=[1, 2, 2, 1])
        print('h_conv1: ', h_conv1)

        # RB2 (36 -> 36)
        h_rb2 = h_conv1
        for resi in range(3):
            h_rb2 = self.resblock(h_rb2, 128)
            print('h_rb2: ', h_rb2)

        # CL2 (36 -> 18)
        h_conv2 = self.conv(h_rb2, 128, 256, strides=[1, 2, 2, 1])
        print('h_conv2: ', h_conv2)

        # RB3 (18 -> 18)
        h_rb3 = h_conv2
        for resi in range(5):
            h_rb3 = self.resblock(h_rb3, 256)
            print('h_rb3: ', h_rb3)

        # CL3 (18 -> 9)
        h_conv3 = self.conv(h_rb3, 256, 512, strides=[1, 2, 2, 1])
        print('h_conv3: ', h_conv3)

        # RB4 (9 -> 9)
        h_rb4 = h_conv3
        for resi in range(2):
            h_rb4 = self.resblock(h_rb4, 512)
            print('h_rb4: ', h_rb4)

        print('-- densely connected layer --')
        W_dl = self.weight_variable([9 * 9 * 512, 1024])
        b_dl = self.bias_variable([1024])
        h_flat = tf.reshape(h_rb4, [self.batch_size, 9 * 9 * 512])
        h_dl = tf.nn.relu(tf.matmul(h_flat, W_dl) + b_dl)
        print('h_dl: ', h_dl)

        W_vec = self.weight_variable([1024, 4])
        b_vec = self.bias_variable([4])
        h_vec = tf.matmul(h_dl, W_vec) + b_vec
        print('h_vec: ', h_vec)

        d_c = tf.reduce_mean(tf.squared_difference(y_[:, :2], h_vec[:, :2]))
        d_r = tf.reduce_mean(tf.squared_difference(y_[:, 3], h_vec[:, 3]))

        angle_gt = y_[:, 2]
        angle_pr = h_vec[:, 2]
        d_a_cw = tf.square(angle_gt - angle_pr)
        d_a_ccw = tf.square(180 - tf.abs(angle_gt - angle_pr))
        d_a = tf.reduce_mean(tf.minimum(d_a_cw, d_a_ccw))

        cost = (d_c + d_r + d_a)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

        tf.add_to_collection('batch_size_', self.batch_size)
        tf.add_to_collection('h_vec_', h_vec)
        tf.add_to_collection('cost_', cost)

        sess.run(tf.global_variables_initializer())

        if not os.path.isdir(self.checkout_dir):
            os.makedirs(self.checkout_dir)

        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(self.checkout_dir)
        if checkpoint:
            print "Restoring from checkpoint", checkpoint
            saver.restore(sess, checkpoint)
        else:
            print "Couldn't find checkpoint to restore from. Starting over."

        fig = utils.init_figure_with_idx(fig_idx=0,
                                         figsize=[4, 4])
        fig_dir = 'fig_rst_v2'
        if os.path.isdir(fig_dir):
            shutil.rmtree(fig_dir)
        os.makedirs(fig_dir)

        mc_train = int(len(train_imgs) / float(self.batch_size))
        mc_test = int(len(test_imgs) / float(self.batch_size))
        for ei in range(500):
            # training
            for ri in range(mc_train):
                this_X = train_imgs[ri * self.batch_size:
                                    ri * self.batch_size +
                                    self.batch_size]
                this_Y = train_vecs[ri * self.batch_size:
                                    ri * self.batch_size +
                                    self.batch_size]
                sess.run(train_step, feed_dict={x: this_X, y_: this_Y})
                if ri % 50 == 0:
                    t_cost, tx, th_vec = sess.run([cost, x, h_vec],
                                                  feed_dict={x: this_X,
                                                             y_: this_Y})
                    print('ep%d, iter %d, dist: %g' % (ei, ri, t_cost))
                    print(th_vec[ei % self.batch_size],
                          this_Y[ei % self.batch_size])

                    saver.save(sess, os.path.join(self.checkout_dir,
                                                  'SP_Layer_1_v2'))
            # test
            avg_dist = 0.0
            for ri in range(mc_test):
                this_X = test_imgs[ri * self.batch_size:
                                   ri * self.batch_size +
                                   self.batch_size]
                this_Y = test_vecs[ri * self.batch_size:
                                   ri * self.batch_size +
                                   self.batch_size]
                t_cost, th_vec = sess.run([cost, h_vec],
                                          feed_dict={x: this_X,
                                                     y_: this_Y})
                avg_dist += t_cost
                if ri % 5 == 0:
                    fo_img_name = 'ep%03d_it%04d.png' % (ei, ri)
                    fig.clf()
                    self.draw_results_plot(this_X[ei % self.batch_size],
                                           th_vec[ei % self.batch_size])
                    utils.save_fig_in_dir(fig, dirname=fig_dir,
                                          filename=fo_img_name)
            print('\n[test ep%d] avg_dist: %f\n' % (ei, (avg_dist / mc_test)))

        sess.close()
        print('train finished')

    def draw_results_plot(self, img, gt, rn=1, cn=1, idx=1):
        center = gt[:2]
        angle = math.radians(gt[2])
        rho = gt[3]

        x = [center[0] - rho * np.cos(angle),
             center[0] + rho * np.cos(angle)]
        y = [center[1] - rho * np.sin(angle),
             center[1] + rho * np.sin(angle)]

        sp = utils.imshow_in_subplot(rn, cn, idx, img.astype(self.img_dtype))

        sp.plot(x, y, '-', color='#40E0D0', linewidth=3.0, alpha=0.8)

        sp.set_ylim(img.shape[0], 0)
        sp.set_xlim(0, img.shape[1])

    def load_data(self):
        if (not os.path.isdir(self.train_dir) or
                not os.path.isdir(self.test_dir)):
            print('could not open base directory')
            return None, None, None, None
        print('load data')

        train_files = fnmatch.filter(os.listdir(self.train_dir), '*.png')
        test_files = fnmatch.filter(os.listdir(self.test_dir), '*.png')

        train_imgs = list()
        train_vecs = list()
        test_imgs = list()
        test_vecs = list()

        for tfile in train_files:
            t_img_name = os.path.join(self.train_dir, tfile)
            t_vec_name = os.path.join(self.train_dir, tfile[:-3] + 'json')
            t_img = misc.imread(t_img_name, flatten=False)
            train_imgs.append(t_img)
            t_symm_axis = json.load(open(t_vec_name))
            t_vec = (t_symm_axis['center'] +
                     [math.degrees(t_symm_axis['angle']), t_symm_axis['rho']])
            train_vecs.append(t_vec)

        for tfile in test_files:
            t_img_name = os.path.join(self.test_dir, tfile)
            t_vec_name = os.path.join(self.test_dir, tfile[:-3] + 'json')
            t_img = misc.imread(t_img_name, flatten=False)
            self.img_dtype = t_img.dtype
            test_imgs.append(t_img)
            t_symm_axis = json.load(open(t_vec_name))
            t_vec = (t_symm_axis['center'] +
                     [math.degrees(t_symm_axis['angle']), t_symm_axis['rho']])
            test_vecs.append(t_vec)

        return train_imgs, train_vecs, test_imgs, test_vecs

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv(self, x, ch_in, ch_out, strides=[1, 1, 1, 1], ksize=3):
        W = self.weight_variable([ksize, ksize, ch_in, ch_out])
        b = self.bias_variable([ch_out])
        h_conv = tf.nn.relu(self.conv2d(x, W, strides=strides) + b)
        return h_conv

    def conv2d(self, x, W, strides=[1, 1, 1, 1]):
        return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

    def resblock(self, x, ch):
        W1 = self.weight_variable([1, 1, ch, ch])
        b1 = self.bias_variable([ch])
        lv1 = tf.nn.relu(self.conv2d(x, W1, strides=[1, 1, 1, 1]) + b1)

        W2 = self.weight_variable([1, 1, ch, ch])
        b2 = self.bias_variable([ch])
        lv2_conv = self.conv2d(lv1, W2, strides=[1, 1, 1, 1]) + b2

        return tf.nn.relu(tf.add(lv2_conv, x))

    def deconv(self, x, ch_in, ch_out, out_shape,
               strides=[1, 1, 1, 1], ksize=3):
        W = self.weight_variable([ksize, ksize, ch_out, ch_in])
        b = self.bias_variable([ch_out])
        h_dconv = tf.nn.relu(self.deconv2d(x, W, out_shape,
                                           strides=strides) + b)
        return h_dconv

    def deconv2d(self, x, W, out_shape, strides=[1, 1, 1, 1]):
        return tf.nn.conv2d_transpose(x, W, output_shape=out_shape,
                                      strides=strides, padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


def script_main():
    print('\ntrain layer 1 - started\n')

    if not os.path.isfile('config.json'):
        print('Cannot open config file.')
        return

    with open('config.json') as config_file:
        config = json.load(config_file)

    if config is None:
        print('Empty config file')
        return

    train_dir = os.path.join(config['base_dir'], config['train_dir'])
    test_dir = os.path.join(config['base_dir'], config['test_dir'])
    checkout_dir = os.path.join(config['base_dir'], config['checkout_l1'])

    deep_symm = SP_Layer_1(train_dir, test_dir, checkout_dir)
    deep_symm.run_train()

    print('\ntrain layer 1 - finished\n')


if __name__ == '__main__':
    script_main()


# end of script
