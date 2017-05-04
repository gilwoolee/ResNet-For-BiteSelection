# strawberry_sampler.py

import numpy as np
from scipy import misc
import os
import sys
import simplejson as json
import shutil
import matplotlib.pyplot as plt
import copy
import math
import fnmatch

from im_modules import utils


fig = None
target_dir = None
cur_image_name = None


def group_l2_data():
    print('group_l2_data')

    train_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/l2_train'
    test_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/l2_test'
    utils.remake_dir(train_dir)
    utils.remake_dir(test_dir)

    data_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/l2_data'
    images = fnmatch.filter(os.listdir(data_dir), '*.png')

    train_count = 0
    test_count = 0
    for idx, imgname in enumerate(images):
        fullname = os.path.join(data_dir, imgname)
        if idx < len(images) * 0.1:
            shutil.copy(fullname, os.path.join(test_dir, imgname))
            shutil.copy(fullname[:-3] + 'out',
                        os.path.join(test_dir, imgname[:-3] + 'out'))
            test_count += 1
        else:
            shutil.copy(fullname, os.path.join(train_dir, imgname))
            shutil.copy(fullname[:-3] + 'out',
                        os.path.join(train_dir, imgname[:-3] + 'out'))
            train_count += 1
    print(train_count, test_count)


def gen_l2_data():
    print('gen_l2_data')

    target_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/l2_data'
    utils.remake_dir(target_dir)

    sample_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/seg_sampling'
    files = os.listdir(sample_dir)

    rot_options = np.linspace(0, 360, 36, endpoint=False)
    flip_options = [False, True]

    for idx, tname in enumerate(files):
        if not tname.endswith('.png'):
            continue

        img_name = os.path.join(sample_dir, tname)
        img_org = misc.imread(img_name)
        img = misc.imresize(img_org, [144, 144, 3])

        gt_file = open(img_name[:-3] + 'out', 'r')
        gt = np.load(gt_file)
        gt_file.close()

        for t_flip in flip_options:
            for t_rot in rot_options:
                t_gt = copy.deepcopy(gt)
                t_img = copy.deepcopy(img)

                if t_flip:
                    t_img = t_img[:, ::-1]
                    t_gt[0] = (t_img.shape[1] -
                               t_gt[0])

                t_img = utils.rotate_in_degrees(t_img, -t_rot)
                rotate_point(t_gt, math.radians(t_rot), t_img.shape)

                outfile_name = '%s_%d_%s' % (tname[:-4], t_rot,
                                             'F' if t_flip else 'N')
                outfile_name = os.path.join(target_dir, outfile_name)

                misc.imsave(outfile_name + '.png', t_img)
                outfile = open(outfile_name + '.out', 'w')
                np.save(outfile, t_gt)
                outfile.close()
                print('created: %s' % outfile_name)


def rotate_point(p, rot_rad, shape):
    s = np.sin(rot_rad)
    c = np.cos(rot_rad)

    p[0] -= shape[1] / 2
    p[1] -= shape[0] / 2

    new_x = p[0] * c - p[1] * s
    new_y = p[0] * s + p[1] * c

    p[0] = new_x + shape[1] / 2
    p[1] = new_y + shape[0] / 2
    return p


def test_sampling():
    print('test_sampling')
    target_dir = 'fig_rst'
    utils.remake_dir('fig_rst')

    sample_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/seg_sampling'
    files = os.listdir(sample_dir)

    fig = utils.init_figure_with_idx(1, figsize=[4, 4])
    for idx, tname in enumerate(files):
        if not tname.endswith('.png'):
            continue

        img_name = os.path.join(sample_dir, tname)
        img_org = misc.imread(img_name)
        img = misc.imresize(img_org, [144, 144, 3])

        gt_file = open(img_name[:-3] + 'out', 'r')
        gt = np.load(gt_file)
        gt_file.close()

        fig.clf()
        draw_results_plot(img, gt)
        utils.save_fig_in_dir(fig, dirname=target_dir,
                              filename=tname)


def draw_results_plot(img, gt, rn=1, cn=1, idx=1):
    center = gt
    rho = 8

    x = [center[0] - rho * np.cos(0),
         center[0] + rho * np.cos(0)]
    y = [center[1] - rho * np.sin(0),
         center[1] + rho * np.sin(0)]

    x2 = [center[0] - rho * np.cos(np.pi / 2),
          center[0] + rho * np.cos(np.pi / 2)]
    y2 = [center[1] - rho * np.sin(np.pi / 2),
          center[1] + rho * np.sin(np.pi / 2)]

    sp = utils.imshow_in_subplot(rn, cn, idx, img)

    sp.plot(x, y, '-', color='#51A39D', linewidth=2.0, alpha=0.8)
    sp.plot(x2, y2, '-', color='#51A39D', linewidth=2.0, alpha=0.8)

    sp.set_ylim(img.shape[0], 0)
    sp.set_xlim(0, img.shape[1])


def sampling_fork_positions():
    global fig, target_dir, cur_image_name

    target_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/seg_sampling'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    sample_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/si_one'
    files = os.listdir(sample_dir)

    for idx, tname in enumerate(files):
        if not tname.endswith('.png'):
            continue

        cur_image_name = tname
        img_name = os.path.join(sample_dir, tname)
        img_org = misc.imread(img_name)
        print(img_name, img_org.shape)

        img = misc.imresize(img_org, [144, 144, 3])

        fig = utils.init_figure_with_idx(1)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        utils.imshow(img)
        plt.show(fig)

    sample_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/seg_g_org'
    files = os.listdir(sample_dir)

    for idx, tname in enumerate(files):
        if not tname.endswith('.png'):
            continue

        cur_image_name = tname
        img_name = os.path.join(sample_dir, tname)
        img_org = misc.imread(img_name)
        print(img_name, img_org.shape)

        img = misc.imresize(img_org, [144, 144, 3])

        fig = utils.init_figure_with_idx(1)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        utils.imshow(img)
        plt.show(fig)

    fig.canvas.mpl_disconnect(cid)


def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    new_gt = np.array([event.xdata, event.ydata])

    outfile_name = os.path.join(target_dir, cur_image_name[:-4] + '.out')
    outfile = open(outfile_name, 'w')
    np.save(outfile, new_gt)
    outfile.close()

    plt.close(fig)


def rotate_si_one():
    sample_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/si_one'
    files = os.listdir(sample_dir)

    target_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/seg_g'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    for idx, tname in enumerate(files):
        if not tname.endswith('.png'):
            continue

        img_name = os.path.join(sample_dir, tname)
        img_org = misc.imread(img_name)
        print(img_name, img_org.shape)

        img = misc.imresize(img_org, [144, 144, 3])

        for rd in range(0, 1, 10):
            timg = utils.rotate_in_degrees(img, rd, scale=1.0)
            misc.imsave(os.path.join(target_dir,
                                     'g_%02d_r%03d.png' % (idx + 43, rd)),
                        timg)


def rotate_segments():
    sample_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/seg_g_org'
    files = os.listdir(sample_dir)

    target_dir = '/mnt/hard-ext/yomkiru/Data/Strawberries/seg_g'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    for idx, tname in enumerate(files):
        if not tname.endswith('.png'):
            continue

        img_name = os.path.join(sample_dir, tname)
        img_org = misc.imread(img_name)
        print(img_name, img_org.shape)

        img = misc.imresize(img_org, [144, 144, 3])

        for rd in range(0, 1, 120):
            timg = utils.rotate_in_degrees(img, rd, scale=1.0)
            misc.imsave(os.path.join(target_dir,
                                     'g_%02d_r%03d.png' % (idx, rd)),
                        timg)


def group_data(l1_g_dir, l1_ng_dir, l1_train_dir, l1_test_dir):
    print('group_data')

    print(l1_g_dir, l1_ng_dir, l1_train_dir, l1_test_dir)

    utils.remake_dir(l1_train_dir)
    utils.remake_dir(os.path.join(l1_train_dir, 'g'))
    utils.remake_dir(os.path.join(l1_train_dir, 'ng'))
    utils.remake_dir(l1_test_dir)
    utils.remake_dir(os.path.join(l1_test_dir, 'g'))
    utils.remake_dir(os.path.join(l1_test_dir, 'ng'))

    g_files = os.listdir(l1_g_dir)
    ng_files = os.listdir(l1_ng_dir)

    for fidx, gf_name in enumerate(g_files):
        if fidx < len(g_files) * 0.1:
            shutil.copy(os.path.join(l1_g_dir, gf_name),
                        os.path.join(l1_test_dir, 'g', gf_name))
        else:
            shutil.copy(os.path.join(l1_g_dir, gf_name),
                        os.path.join(l1_train_dir, 'g', gf_name))

    for fidx, ngf_name in enumerate(ng_files):
        if fidx < len(ng_files) * 0.1:
            shutil.copy(os.path.join(l1_ng_dir, ngf_name),
                        os.path.join(l1_test_dir, 'ng', ngf_name))
        else:
            shutil.copy(os.path.join(l1_ng_dir, ngf_name),
                        os.path.join(l1_train_dir, 'ng', ngf_name))


def print_usage():
    print('usage: python strawberry_sampler.py <option>')
    print('options: data_gen, group, sampling, test')
    print('\n')


def read_config():
    if not os.path.isfile('config.json'):
        print('Cannot open config file.')
        return

    with open('config.json') as config_file:
        config = json.load(config_file)

    return config


def script_main():
    if len(sys.argv) != 2:
        print_usage()
        return

    config = read_config()
    if config is None:
        print('Empty config file: check ./config.json')
        return

    print('strawberry_sampler')

    base_dir = config['base_dir']
    l1_g_dir = os.path.join(base_dir, config['layer1_g_dir'])
    l1_ng_dir = os.path.join(base_dir, config['layer1_ng_dir'])
    l1_train_dir = os.path.join(base_dir, config['layer1_train_dir'])
    l1_test_dir = os.path.join(base_dir, config['layer1_test_dir'])

    if sys.argv[1] == 'data_gen':
        rotate_si_one()
        rotate_segments()
    elif sys.argv[1] == 'group':
        group_data(l1_g_dir, l1_ng_dir, l1_train_dir, l1_test_dir)
    elif sys.argv[1] == 'sampling':
        sampling_fork_positions()
    elif sys.argv[1] == 'test':
        test_sampling()
    elif sys.argv[1] == 'gen_l2':
        gen_l2_data()
    elif sys.argv[1] == 'group_l2':
        group_l2_data()
    else:
        print('error: invalid option')
        print_usage()


if __name__ == '__main__':
    script_main()


# end of script
