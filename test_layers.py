# test_layers.py

import numpy as np
from SP_Layer_1 import SP_Layer_1
from SP_Layer_2 import SP_Layer_2
from scipy import misc
import matplotlib.patches as patches

from im_modules import utils


def add_cross_line(sp, gt, rn=1, cn=1, idx=1):
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

    sp.plot(x, y, '-', color='#FFFF00', linewidth=2.0, alpha=0.8)
    sp.plot(x2, y2, '-', color='#FFFF00', linewidth=2.0, alpha=0.8)


def draw_results_plot(img, gt, rn=1, cn=1, idx=1):
    sp = utils.imshow_in_subplot(rn, cn, idx, img)
    sp.set_ylim(img.shape[0], 0)
    sp.set_xlim(0, img.shape[1])

    add_cross_line(sp, gt)


class SlidingBox:
    def __init__(self, ri, ci):
        self.ri = ri
        self.ci = ci


# -------------------------------------------------

layer1 = SP_Layer_1()

print('test_sample')

test_img = misc.imread('img_sample/test_1.png')
rmax, cmax, _ = test_img.shape
rsize, csize = 110, 110

box_list = list()
img_list_1 = list()
for ri in range(0, rmax - rsize + 20, 20):
    for ci in range(0, cmax - csize + 20, 20):
        t_img = test_img[ri:ri + rsize, ci:ci + csize]
        img_list_1.append(t_img)
        box_list.append(SlidingBox(ri, ci))

rst_1 = layer1.run_network(img_list_1)

rst_dir_l1 = 'l1_rst'
utils.remake_dir(rst_dir_l1)

fig1 = utils.init_figure_with_idx(0, figsize=[10, 6])
sp1 = utils.imshow_in_subplot(1, 1, 1, test_img)

img_list_2 = list()
box_list_2 = list()
for idx in range(len(img_list_1)):
    if rst_1[idx][0] > rst_1[idx][1]:  # good
        img_list_2.append(img_list_1[idx])
        box_list_2.append(box_list[idx])
        sp1.add_patch(
            patches.Rectangle(
                (box_list[idx].ci, box_list[idx].ri),
                csize, rsize,
                linewidth=1,
                edgecolor='b',
                facecolor='none'
            )
        )

layer1 = None

# -------------------------------------------------

layer2 = SP_Layer_2()

rst_2 = layer2.run_network(img_list_2)

rst_dir_l2 = 'l2_rst'
utils.remake_dir(rst_dir_l2)

fig2 = utils.init_figure_with_idx(1, figsize=[5, 5])

for idx in range(len(img_list_2)):
    t_rst = rst_2[idx]
    t_img = img_list_2[idx]

    t_img = utils.flatten_image(t_img)

    t_box = box_list_2[idx]
    relative_rst = [t_box.ci + (t_rst[0] / 144. * csize),
                    t_box.ri + (t_rst[1] / 144. * rsize)]

    fig2.clf()
    draw_results_plot(t_img, t_rst)
    utils.save_fig_in_dir(fig2, dirname=rst_dir_l2,
                          filename='l2_%03d.png' % idx)
    add_cross_line(sp1, relative_rst)


utils.save_fig_in_dir(fig1, dirname=rst_dir_l1,
                      filename='l1_rst.png')

# end of script
