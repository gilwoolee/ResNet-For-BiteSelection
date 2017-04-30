# strawberry_sampler.py

import numpy as np
from scipy import misc
import os
import sys
import simplejson as json
import shutil
import matplotlib.pyplot as plt

from im_modules import utils


fig = None
target_dir = None
cur_image_name = None


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
    else:
        print('error: invalid option')
        print_usage()


if __name__ == '__main__':
    script_main()


# end of script
