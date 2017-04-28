""" utils.py

Image utility methods
by Youngsun Kim
Oct 2016
"""

import numpy as np
from scipy.signal import fftconvolve
import scipy.ndimage as ndi
import scipy.ndimage.interpolation as itp
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import cv2
import sys


def get_interpolants(sig, mul=10):
    x = np.linspace(0, len(sig) - 1, num=len(sig), endpoint=True)
    y = sig
    f2 = interpolate.interp1d(x, y, kind='quadratic')
    xnew = np.linspace(0, len(sig) - 1, num=len(sig) * mul, endpoint=True)
    return f2(xnew)


def rebin(a, shape, method='mean'):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    if method == 'max':
        return a.reshape(sh).max(-1).max(1)
    else:
        return a.reshape(sh).mean(-1).mean(1)


def crop_image(img, margin=(80, 80, 80, 80)):
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (margin[0], margin[1],
            img.shape[1] - margin[0] - margin[2],
            img.shape[0] - margin[1] - margin[3])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return img * mask[:, :, np.newaxis]


def get_resized_image(img, size=0):
    img_width, img_height = img.shape
    if size > 0:
        max_size = size
    else:
        if img_width >= 169:
            max_size = 169
        elif img_width >= 121:
            max_size = 121
        elif 49 <= img_width < 70:
            max_size = 49
        elif img_width % 2 == 0:
            max_size = img_width + 1
        else:
            max_size = img_width
    img_height = int(np.round(img_height * (float(max_size) / img_width)))
    img_width = max_size
    return cv2.resize(img, (img_width, img_height))


def flatten_image(img):
    if len(img.shape) > 2:
        return img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    else:
        return img


def get_wconv_gray(img, mw):
    wc = np.zeros_like(img, dtype=np.complex128)
    for tmw in mw:
        wc += abs(fft_convolve(img, tmw))
    return wc


def get_gaussian_mask(rmax, cmax):
    k_rows = np.floor(rmax / 2.)
    k_cols = np.floor(cmax / 2.)

    sigma = rmax / 6.
    x, y = np.meshgrid(
        np.arange(-k_rows, k_rows + 1),
        np.arange(-k_cols, k_cols + 1))
    x, y = x.astype(np.float), y.astype(np.float)
    u2 = (x ** 2) + (y ** 2)
    return np.exp(-u2 / (2. * (sigma ** 2)))


def get_circle_mask(rmax, cmax, dim_ratio=1.0):
    mask = np.zeros((rmax, cmax))
    k_row = int(rmax * 0.5)
    k_col = int(cmax * 0.5)
    dim = int(min(k_row, k_col) * 0.87 * dim_ratio)
    margin_row = int(k_row * 0.3)
    margin_col = int(k_col * 0.3)
    if margin_row % 2 == 0:
        margin_row -= 1
    if margin_col % 2 == 0:
        margin_col -= 1
    if is_cv2():
        cv2.circle(mask, (k_col, k_row), dim, 1, -1)  # opencv 2
    else:
        mask = cv2.circle(mask, (k_col, k_row), dim, 1, -1)  # opencv 3
    return ndi.filters.gaussian_filter(mask, 3)


def get_masked_image(img, mask, bval=0):
    rmax, cmax = img.shape
    ones = np.ones((rmax, cmax))
    gray_field = ones * bval
    img_masked = img * mask
    gray_field = gray_field * (ones - mask)
    return img_masked + gray_field


def get_masked_image_gray(img, mask):
    rmax, cmax = img.shape
    ones = np.ones((rmax, cmax))
    gray_field = ones * 128
    img_masked = img * mask
    gray_field = gray_field * (ones - mask)
    return img_masked + gray_field


def get_pc(sig_0, sig_1):
    sig_0_c = np.conj(sig_0)
    return sig_1 * sig_0_c


def take_ipc_from_signals(sig_0, sig_1):
    return np.fft.ifft(get_pc(np.fft.fft(sig_0), np.fft.fft(sig_1)))


def take_ipc(sig_0, sig_1):
    return np.fft.ifft(get_pc(sig_0, sig_1))


def take_ipc2_from_images(sig_0, sig_1):
    sig_0 = take_fft2_shifted(sig_0)
    sig_1 = take_fft2_shifted(sig_1)
    return take_ipc2(sig_0, sig_1)


def take_ipc2(sig_0, sig_1):
    return np.fft.ifftshift(np.fft.ifft2(get_pc(sig_0, sig_1)))


def take_fft(sig):
    return np.fft.fft(sig)


def take_fft_shifted(sig):
    return np.fft.fftshift(np.fft.fft(sig))


def take_fft2(input_img, mul=1):
    return np.fft.fft2(input_img,
                       [int(input_img.shape[0] / mul), int(input_img.shape[1] / mul)])


def take_fft2_shifted(input_img, mul=1):
    return np.fft.fftshift(take_fft2(input_img, mul))


def fft_convolve(img, kernel):
    pad_row = int(kernel.shape[0] / 2.0)
    pad_col = int(kernel.shape[1] / 2.0)
    timg = np.pad(img, ((pad_row, pad_row), (pad_col, pad_col)), mode='edge')
    rst = fftconvolve(timg, kernel, mode='same')
    return rst[pad_row:pad_row + img.shape[0], pad_col:pad_col + img.shape[1]]


def generate_rs(img_shape):
    _, _, a_base = get_logpolar_base(img_shape)
    rs, _ = np.meshgrid(np.arange(0, img_shape[1]), np.zeros(img_shape[0]))
    rs = rs * np.log(a_base)  # r' * log a = r
    return np.exp(2 * rs), a_base


def rotate_in_degrees(img, deg, scale=1.0):
    if len(img.shape) == 2:
        return rotate_in_degrees_gray(img, deg, scale)
    else:
        timg = np.zeros_like(img)
        timg[:, :, 0] = rotate_in_degrees_gray(img[:, :, 0], deg, scale)
        timg[:, :, 1] = rotate_in_degrees_gray(img[:, :, 1], deg, scale)
        timg[:, :, 2] = rotate_in_degrees_gray(img[:, :, 2], deg, scale)
        return timg


def rotate_in_degrees_gray(img, deg, scale=1.0):
    centre = 0.5 * np.array(img.shape)
    theta = math.radians(deg)
    rot = np.array([[np.cos(theta) / scale, np.sin(theta)],
                    [-np.sin(theta), np.cos(theta) / scale]])
    offset = (centre - centre.dot(rot)).dot(np.linalg.inv(rot))
    return itp.affine_transform(
        img, rot, order=1, offset=-offset, cval=0.0,
        output=np.float32, mode='nearest')


def rotate_in_radians(img, theta, scale=1.0):
    centre = 0.5 * np.array(img.shape)
    rot = np.array([[np.cos(theta) / scale, np.sin(theta)],
                    [-np.sin(theta), np.cos(theta) / scale]])
    offset = (centre - centre.dot(rot)).dot(np.linalg.inv(rot))

    if (img.dtype == np.dtype('complex64') or
            img.dtype == np.dtype('complex128')):
        out_real = itp.affine_transform(img.real, rot, order=1, offset=-offset,
                                        cval=0.0, output=np.float32,
                                        mode='nearest')
        out_imag = itp.affine_transform(img.imag, rot, order=1, offset=-offset,
                                        cval=0.0, output=np.float32,
                                        mode='nearest')
        output = out_real + 1j * out_imag
    else:
        output = itp.affine_transform(img, rot, order=1, offset=-offset,
                                      cval=0.0, output=np.float32,
                                      mode='nearest')
    return output


def get_a_base(shape):
    maxd = min(shape[0], shape[1])
    return (maxd / 2.) ** (1. / maxd)


def get_logpolar_base(shape):
    theta = -1 * np.linspace(0, np.pi * 2, shape[0], endpoint=False)
    theta = np.roll(theta, -1 * int(shape[0] * 0.25))
    a_base = get_a_base(shape)
    radius = (a_base ** np.arange(shape[1], dtype=np.float64))
    return theta, radius, a_base


def get_rs(shape, a_base):
    rs, _ = np.meshgrid(np.arange(shape[1]), np.zeros(shape[0]))
    rs = rs * np.log(a_base)  # r' * log a = r
    return np.exp(2. * rs)


def polar(image):
    theta = np.empty_like(image, dtype=np.float64)
    radius = np.arange(image.shape[1], dtype=np.float64) + 2
    theta.T[:], _, a_base = get_logpolar_base(image.shape)
    x = radius * np.sin(theta) + image.shape[0] / 2
    y = radius * np.cos(theta) + image.shape[1] / 2
    output = np.empty_like(x)
    itp.map_coordinates(image, [x, y], output=output, order=1)
    return output


def polar_complex(image):
    p_real = polar(image.real)
    p_imag = polar(image.imag)
    return p_real + p_imag * 1j


def logpolar_complex(image, rs=None):
    lp_real, _, _ = logpolar(image.real)
    lp_imag, _, _ = logpolar(image.imag)
    res = lp_real + lp_imag * 1j
    if rs is not None:
        res = res * rs
    return res


def logpolar_normal(image, rs):
    lps, _, _ = logpolar(image)
    return lps * rs


def logpolar(image):
    theta = np.empty_like(image, dtype=np.float64)
    radius = np.empty_like(theta)
    theta.T[:], radius[:], a_base = get_logpolar_base(image.shape)
    x = radius * np.sin(theta) + image.shape[0] / 2
    y = radius * np.cos(theta) + image.shape[1] / 2
    output = np.empty_like(x)
    itp.map_coordinates(image, [x, y], output=output, order=1)
    return output, radius[0], a_base


def logpolar_inverse(lp_image):
    xs, ys = np.meshgrid(
        np.linspace(0, lp_image.shape[0], lp_image.shape[0] + 1),
        np.linspace(0, lp_image.shape[0], lp_image.shape[0] + 1)
    )
    _, _, a_base = get_logpolar_base(lp_image)
    rho = np.log(np.sqrt(xs * xs + ys * ys))
    phi = np.arctan2(xs, ys)
    output = np.empty_like(rho)
    itp.map_coordinates(lp_image, [rho, phi], output=output)
    return output


def get_argmax_2d(sig):
    am_t = np.argmax(sig)
    t_row, t_col = np.unravel_index(am_t, sig.shape)
    t_row -= np.ceil(sig.shape[0] / 2.)
    t_col -= np.ceil(sig.shape[1] / 2.)
    if sig.shape[0] % 2 == 0:
        t_row -= 1
    if sig.shape[1] % 2 == 0:
        t_col -= 1
    return [t_row, t_col]


def draw_rectangle(sp, offset, width, height):
    sp.add_patch(
        patches.Rectangle(
            (float(offset[0]), float(offset[1])), width - 1, height - 1,
            linewidth=1.5, edgecolor='red', fill=False
        )
    )


def imshow_in_subplot_with_labels(r, c, idx, img, xlabel, ylabel):
    sp = plt.subplot(r, c, idx)
    sp.set_xlabel(xlabel)
    sp.set_ylabel(ylabel)
    imshow_gray(img)
    return sp


def imshow_in_subplot_with_title(r, c, idx, img, title):
    sp = plt.subplot(r, c, idx)
    sp.set_title(title)
    imshow_gray(img)
    return sp


def imshow_in_subplot(r, c, idx, img):
    sp = plt.subplot(r, c, idx)
    imshow_gray(img)
    return sp


def imshow(img):
    plt.imshow(img, interpolation='nearest')


def imshow_gray(img):
    plt.imshow(img, cmap='gray', interpolation='nearest')


def plot_in_subplot_with_title(r, c, idx, plot, title, color='b'):
    sp = plt.subplot(r, c, idx)
    sp.set_title(title)
    plot_with_margin(plot, color)
    return sp


def plot_in_subplot(r, c, idx, plot, color='b'):
    sp = plt.subplot(r, c, idx)
    plot_with_margin(plot, color)
    return sp


def plot_with_margin(plot, color='b'):
    if color is None:
        color = 'b'
    plt.plot(plot, color=color)
    margin = len(plot) * 0.1
    plt.xlim(-margin, len(plot) + margin)


def init_figure(fig_idx):
    fig_idx += 1
    this_fig = plt.figure(fig_idx)
    this_fig.set_tight_layout(True)
    return this_fig


def init_figure_with_idx(fig_idx, figsize=(10, 10)):
    this_fig = plt.figure(fig_idx, figsize=figsize)
    this_fig.set_tight_layout(True)
    return this_fig


def show_plots():
    plt.show()


def save_fig_in_dir(fig, dirname='', filename=None):
    if filename is None:
        return

    import os
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    fig.savefig(os.path.join(dirname, filename))


def open_in_preview(file_names):  # for mac
    params = ['open', '-a', 'Preview']
    import subprocess
    subprocess.call(params.extend(file_names))


def is_cv2():
    return check_opencv_version("2.")


def is_cv3():
    return check_opencv_version("3.")


def check_opencv_version(major, lib=None):
    if lib is None:
        import cv2 as lib
    return lib.__version__.startswith(major)


def is_python2():
    return python_version() == 2


def is_python3():
    return python_version() == 3


def python_version():
    return sys.version_info.major

# End of script
