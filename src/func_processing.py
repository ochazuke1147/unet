import cv2
import numpy as np
from src.normalize import *

IMAGE_SIZE = 128


def equalize_hist_masked(gray_image, mask):
    hist_mask, bins_mask = np.histogram(mask.ravel(), 256, [0, 256])
    height, width = gray_image.shape[0], gray_image.shape[1]
    sum_pixel = hist_mask[-1]
    masked = cv2.bitwise_and(gray_image, mask)
    imax = masked.max()
    hist, bins = np.histogram(masked.ravel(), 256, [0, 256])
    hist[0] -= hist_mask[0]
    dst = np.empty((height, width))
    for y in range(0, height):
        for x in range(0, width):
            #print(mask[y][x], gray_image[y][x])
            if mask[y][x] == 255:
                dst[y][x] = np.sum(hist[0: gray_image[y][x]]) * (imax / sum_pixel)
                #print(mask[y][x], gray_image[y][x] ,dst[y][x])
            else:
                dst[y][x] = 255
    dst = np.uint8(dst)

    return dst


def highlight_vein(gray_image, masking=False, mask=None):
    kernel_size = 15
    kernel = np.full((kernel_size, kernel_size), -1, dtype=np.float)
    kernel[7, 7] = 225

    if masking:
        image = equalize_hist_masked(gray_image, mask)
    else:
        image = cv2.equalizeHist(gray_image)
    image = cv2.filter2D(image, -1, kernel)
    image = cv2.medianBlur(image, 5)
    #image = cv2.medianBlur(image, 5)

    return image

def unet_masking(gray_image):
    from src.unet import UNet
    size = (gray_image.shape[1], gray_image.shape[0])
    images = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    image = cv2.resize(gray_image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image[:, :, np.newaxis]
    #images[0] = normalize_x(image)
    images[0] = image

    input_channel_count = 1
    output_channel_count = 1
    first_layer_filter_count = 64
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.load_weights('unet_weights.hdf5')
    BATCH_SIZE = 1

    Y_pred = model.predict(images, BATCH_SIZE)
    y = cv2.resize(Y_pred[0], size)
    y_dn = denormalize_y(y)
    y_dn = np.uint8(y_dn)
    ret, mask = cv2.threshold(y_dn, 0, 255, cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(gray_image, mask)
    mask_rest = cv2.bitwise_not(mask)
    masked = cv2.bitwise_or(masked, mask_rest)

    return masked


def segnet_masking(gray_image):
    from src.segnet import segnet
    from src.timer import Timer

    timer = Timer()
    size = (gray_image.shape[1], gray_image.shape[0])
    images = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    image = cv2.resize(gray_image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image[:, :, np.newaxis]
    #images[0] = normalize_x(image)
    images[0] = image

    input_channel_count = 1
    output_channel_count = 1
    first_layer_filter_count = 64
    model = segnet(input_channel_count, output_channel_count, first_layer_filter_count)
    timer.time_elapsed()
    model.load_weights('segnet_weights.hdf5')
    timer.time_elapsed()
    BATCH_SIZE = 1

    Y_pred = model.predict(images, BATCH_SIZE)
    timer.time_elapsed()
    y = cv2.resize(Y_pred[0], size)
    y_dn = denormalize_y(y)
    y_dn = np.uint8(y_dn)
    #cv2.imshow('', y_dn)
    #cv2.waitKey()
    ret, mask = cv2.threshold(y_dn, 0, 255, cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    mask_rest = cv2.bitwise_not(mask)
    masked = cv2.bitwise_or(masked, mask_rest)

    return mask, masked


def opening_masking(gray_image):
    from src.timer import Timer

    timer = Timer()
    cv2.imwrite('thesis/original.png', gray_image)
    kernel = np.ones((15, 15), np.uint8)
    tmp_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,19)), iterations=5)
    cv2.imwrite('thesis/opening.png', tmp_image)
    timer.time_elapsed()

    #tmp_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel, iterations=10)
    #tmp_image = cv2.morphologyEx(gray_image, cv2.MORPH_ERODE, kernel, iterations=10)
    #tmp_image = cv2.morphologyEx(tmp_image, cv2.MORPH_DILATE, kernel, iterations=10)

    if len(tmp_image.shape) == 3:
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)
        print(0)

    cv2.equalizeHist(tmp_image, tmp_image)
    cv2.imwrite('thesis/equalized.png', tmp_image)

    timer.time_elapsed()

    ret, mask = cv2.threshold(tmp_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite('thesis/binary.png', mask)

    masked = cv2.bitwise_and(gray_image, mask)
    mask_rest = cv2.bitwise_not(mask)
    masked = cv2.bitwise_or(masked, mask_rest)
    cv2.imwrite('masked.png', masked)

    cv2.imshow('', masked)
    cv2.waitKey()

    return mask, masked


def compare_images(img1, img2):
    img_or = cv2.bitwise_or(img1, img2)

    cv2.imshow('', img_or)
    cv2.waitKey()
    cv2.imwrite('or.png', img_or)
