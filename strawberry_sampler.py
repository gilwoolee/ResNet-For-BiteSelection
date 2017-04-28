# strawberry_sampler.py

from scipy import misc
import os
import sys
import simplejson as json
import shutil

from im_modules import utils


def rotate_si_one():
    sample_dir = 'si_one'
    files = os.listdir(sample_dir)

    for idx, tname in enumerate(files):
        if not tname.endswith('.png'):
            continue

        img_name = os.path.join(sample_dir, tname)
        img_org = misc.imread(img_name)
        print(img_name, img_org.shape)

        img = misc.imresize(img_org, [144, 144, 3])

        for rd in range(0, 360, 10):
            timg = utils.rotate_in_degrees(img, rd, scale=1.0)
            misc.imsave(os.path.join('seg_g',
                                     'g_%02d_r%03d.png' % (idx + 43, rd)),
                        timg)


def rotate_segments():
    sample_dir = 'seg_ng_org'
    files = os.listdir(sample_dir)

    for idx, tname in enumerate(files):
        if not tname.endswith('.png'):
            continue

        img_name = os.path.join(sample_dir, tname)
        img_org = misc.imread(img_name)
        print(img_name, img_org.shape)

        img = misc.imresize(img_org, [144, 144, 3])

        for rd in range(0, 360, 120):
            timg = utils.rotate_in_degrees(img, rd, scale=1.0)
            misc.imsave(os.path.join('seg_ng',
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
    print('options: data_gen, group')
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
    elif sys.argv[1] == 'group':
        group_data(l1_g_dir, l1_ng_dir, l1_train_dir, l1_test_dir)
    else:
        print('error: invalid option')
        print_usage()


if __name__ == '__main__':
    script_main()


# end of script
