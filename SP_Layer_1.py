# SP_Layer_1.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy import misc
import json
import fnmatch
import numpy as np
import random
import sys
import tensorflow as tf


class SP_Layer_1():
    def __init__(self, config_filename='config.json', batch_size=20):
        if not os.path.isfile('config.json'):
            print('Cannot open config file.')
            return

        with open('config.json') as config_file:
            config = json.load(config_file)

        if config is None:
            print('Empty config file')
            return

        base_dir = config['base_dir']
        train_dir = os.path.join(base_dir, config['layer1_train_dir'])
        test_dir = os.path.join(base_dir, config['layer1_test_dir'])
        checkout_dir = os.path.join(base_dir, config['checkout_l1'])

        self.train_dir_g = os.path.join(train_dir, 'g')
        self.train_dir_ng = os.path.join(train_dir, 'ng')
        self.test_dir_g = os.path.join(test_dir, 'g')
        self.test_dir_ng = os.path.join(test_dir, 'ng')

        self.checkout_dir = checkout_dir
        self.batch_size = batch_size

        self.sess = None

    def isInitialized(self):
        if self.checkout_dir is not None and len(self.checkout_dir) > 0:
            return True
        else:
            return False

    def preprocess_images(self, img_list):
        for idx in range(len(img_list)):
            t_img = misc.imresize(img_list[idx], [144, 144, 3])
            t_img = t_img / 255.
            t_img = t_img - np.mean(t_img)
            img_list[idx] = t_img

    def run_network(self, img_list=None):
        print('network started')

        is_test = False
        if img_list is not None:
            is_test = True
            self.batch_size = len(img_list)
            self.preprocess_images(img_list)
        else:
            train_imgs, train_vecs, test_imgs, test_vecs = self.load_data()
            if train_imgs is None or test_imgs is None:
                return

        tf.reset_default_graph()

        img_res = 144
        x = tf.placeholder(tf.float32, shape=[None, img_res, img_res, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        # CL0 (144 -> 72)
        h_conv0 = self.conv(x, 3, 64, strides=[1, 2, 2, 1])

        # RB1 (72 -> 72)
        h_rb1 = h_conv0
        for resi in range(1):  # 3
            h_rb1 = self.resblock(h_rb1, 64)

        # CL1 (72 -> 36)
        h_conv1 = self.conv(h_rb1, 64, 128, strides=[1, 2, 2, 1])

        # RB2 (36 -> 36)
        h_rb2 = h_conv1
        for resi in range(1):  # 3
            h_rb2 = self.resblock(h_rb2, 128)

        # CL2 (36 -> 18)
        h_conv2 = self.conv(h_rb2, 128, 256, strides=[1, 2, 2, 1])

        # RB3 (18 -> 18)
        h_rb3 = h_conv2
        for resi in range(2):  # 5
            h_rb3 = self.resblock(h_rb3, 256)

        # CL3 (18 -> 9)
        h_conv3 = self.conv(h_rb3, 256, 512, strides=[1, 2, 2, 1])

        # RB4 (9 -> 9)
        h_rb4 = h_conv3
        for resi in range(1):  # 2
            h_rb4 = self.resblock(h_rb4, 512)

        W_dl = self.weight_variable([9 * 9 * 512, 1024])
        b_dl = self.bias_variable([1024])
        h_flat = tf.reshape(h_rb4, [self.batch_size, 9 * 9 * 512])
        h_dl = tf.nn.relu(tf.matmul(h_flat, W_dl) + b_dl)

        W_vec = self.weight_variable([1024, 2])
        b_vec = self.bias_variable([2])
        h_vec = tf.matmul(h_dl, W_vec) + b_vec

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_vec))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(h_vec, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()

        # if is_test:
        #     saver = tf.train.import_meta_graph(os.path.join(self.checkout_dir,
        #                                                     'SP_Layer_1.meta'))
        # else:
        saver = tf.train.Saver()

        if not os.path.isdir(self.checkout_dir):
            os.makedirs(self.checkout_dir)

        checkpoint = tf.train.latest_checkpoint(self.checkout_dir)
        if checkpoint:
            print "Restoring from checkpoint", checkpoint
            saver.restore(self.sess, checkpoint)
        else:
            print "Couldn't find checkpoint to restore from. Starting over."
            self.sess.run(tf.global_variables_initializer())

        if is_test:
            th_vec = self.sess.run(h_vec, feed_dict={x: img_list})
            self.sess.close()
            return th_vec

        else:
            # fig = utils.init_figure_with_idx(fig_idx=0,
            #                                  figsize=[4, 4])
            # fig_dir = 'fig_rst_v2'
            # if os.path.isdir(fig_dir):
            #     shutil.rmtree(fig_dir)
            # os.makedirs(fig_dir)

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
                    self.sess.run(train_step,
                                  feed_dict={x: this_X, y_: this_Y})
                    if ri % 10 == 0:
                        t_cost, th_vec = self.sess.run([cost, h_vec],
                                                       feed_dict={x: this_X,
                                                                  y_: this_Y})
                        print('ep%d, iter %d, cost: %g' % (ei, ri, t_cost))
                        print(th_vec[ei % self.batch_size],
                              this_Y[ei % self.batch_size])
                        outfile = open('./logs/log_layer_1_train.txt', 'a+')
                        outfile.write('%d, %d, %g\n' % (ei, ri, t_cost))
                        outfile.close()

                        saver.save(self.sess, os.path.join(self.checkout_dir,
                                                           'SP_Layer_1'))
                        # test
                        avg_test_cost = 0
                        for ri in range(mc_test):
                            this_X = test_imgs[ri * self.batch_size:
                                               ri * self.batch_size +
                                               self.batch_size]
                            this_Y = test_vecs[ri * self.batch_size:
                                               ri * self.batch_size +
                                               self.batch_size]
                            t_cost, th_vec = self.sess.run([cost, h_vec],
                                                           feed_dict={x: this_X,
                                                                      y_: this_Y})
                            avg_test_cost += t_cost
                        avg_test_cost /= mc_test

                        print('[test] avg_accuracy: %g' % (avg_test_cost))
                        outfile = open('./logs/log_layer_1_test.txt', 'a+')
                        outfile.write('%d, %g\n' % (ei, avg_test_cost))
                        outfile.close()

            print('train finished')
            self.sess.close()
            return

    # def draw_results_plot(self, img, gt, rn=1, cn=1, idx=1):
    #     center = gt[:2]
    #     angle = math.radians(gt[2])
    #     rho = gt[3]

    #     x = [center[0] - rho * np.cos(angle),
    #          center[0] + rho * np.cos(angle)]
    #     y = [center[1] - rho * np.sin(angle),
    #          center[1] + rho * np.sin(angle)]

    #     sp = utils.imshow_in_subplot(rn, cn, idx, img.astype(self.img_dtype))

    #     sp.plot(x, y, '-', color='#40E0D0', linewidth=3.0, alpha=0.8)

    #     sp.set_ylim(img.shape[0], 0)
    #     sp.set_xlim(0, img.shape[1])

    def load_data(self):
        if (not os.path.isdir(self.train_dir_g) or
                not os.path.isdir(self.train_dir_ng) or
                not os.path.isdir(self.test_dir_g) or
                not os.path.isdir(self.test_dir_ng)):
            print('could not open base directory')
            return None, None, None, None
        print('load data')

        train_g_files = fnmatch.filter(os.listdir(self.train_dir_g), '*.png')
        train_ng_files = fnmatch.filter(os.listdir(self.train_dir_ng), '*.png')

        test_g_files = fnmatch.filter(os.listdir(self.test_dir_g), '*.png')
        test_ng_files = fnmatch.filter(os.listdir(self.test_dir_ng), '*.png')

        train_imgs = list()
        train_vecs = list()
        test_imgs = list()
        test_vecs = list()

        for tfile in train_g_files:
            t_img_name = os.path.join(self.train_dir_g, tfile)
            t_img = misc.imread(t_img_name)
            t_img = t_img / 255.
            t_img = t_img - np.mean(t_img)
            train_imgs.append(t_img)
            train_vecs.append([1, 0])

        for tfile in train_ng_files:
            t_img_name = os.path.join(self.train_dir_ng, tfile)
            t_img = misc.imread(t_img_name)
            t_img = t_img / 255.
            t_img = t_img - np.mean(t_img)
            train_imgs.append(t_img)
            train_vecs.append([0, 1])

        combined = list(zip(train_imgs, train_vecs))
        random.shuffle(combined)
        train_imgs[:], train_vecs[:] = zip(*combined)

        for tfile in test_g_files:
            t_img_name = os.path.join(self.test_dir_g, tfile)
            t_img = misc.imread(t_img_name)
            t_img = t_img / 255.
            t_img = t_img - np.mean(t_img)
            test_imgs.append(t_img)
            test_vecs.append([1, 0])

        for tfile in test_ng_files:
            t_img_name = os.path.join(self.test_dir_ng, tfile)
            t_img = misc.imread(t_img_name)
            t_img = t_img / 255.
            t_img = t_img - np.mean(t_img)
            test_imgs.append(t_img)
            test_vecs.append([0, 1])

        combined = list(zip(test_imgs, test_vecs))
        random.shuffle(combined)
        test_imgs[:], test_vecs[:] = zip(*combined)

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


def run_test_sample(deep_symm):
    print('test_sample')
    img_list = list()

    img_list.append(misc.imread('img_sample/l1_good.png'))
    img_list.append(misc.imread('img_sample/l1_not_good.png'))

    print(deep_symm.run_network(img_list=img_list))


def script_main():

    is_test = False
    if len(sys.argv) > 1:
        if len(sys.argv) == 2 and sys.argv[1] == 'test':
            is_test = True
        else:
            print('usage:')
            print('\tfor test  --> python SP_Layer_1.py test')
            print('\tfor train --> python SP_Layer_1.py')
            print('\n')
            return

    deep_symm = SP_Layer_1()

    print('\nSP Layer 1 - started\n')

    if deep_symm.isInitialized:
        if is_test:
            run_test_sample(deep_symm)
        else:
            deep_symm.run_network()

    print('\nSP Layer 1 - finished\n')


if __name__ == '__main__':
    script_main()


# end of script
