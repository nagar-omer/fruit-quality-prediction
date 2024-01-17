import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


def read_sample(path, im_height=323, im_width=323):
    """
    Read image from path and convert it to array of size (size, size, 3)
    """

    # read image + remove edges
    sample = cv2.imread(path)[1:, 1:, :]

    n_rows, n_cols = sample.shape[0] // im_height, sample.shape[1] // im_width
    # split image to n_rows x n_cols
    ims = []
    for row in np.array_split(sample, n_rows):
        for im in np.array_split(row, n_cols, axis=1):
            # drop edges
            ims.append(im[1:-1, 1:-1, :])

    return np.stack(ims, axis=0)


def mask_circle(img, center, radius, out_shape=None):
    assert (center[0] - radius >= 0) and (center[1] - radius >= 0), "circle out of bounds"
    assert (center[0] + radius < img.shape[0]) and (center[1] + radius < img.shape[1]), "circle out of bounds"

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = cv2.circle(mask, center, radius, 255, -1)

    masked_im = cv2.bitwise_and(img, img, mask=mask)

    # crop by center
    if out_shape is not None:
        masked_im = masked_im[center[1] - out_shape[1] // 2:center[1] + out_shape[1] // 2,
                              center[0] - out_shape[0] // 2:center[0] + out_shape[0] // 2]
    return masked_im


def dct_2d(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return dct(dct(img.T, norm='ortho').T, norm='ortho')


def idct_2d(freq):
    return idct(idct(freq.T, norm='ortho').T, norm='ortho')


def apply_edge_detection(img):
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5, 25)
    return edges


if __name__ == '__main__':
    read_sample("/Users/omernagar/Documents/Projects/fruit-quality-prediction/data/A/collage_20230619_165518_l1w0002r186f0_stream00.png")