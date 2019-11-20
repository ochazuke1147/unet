import cv2
import numpy as np
from src.unet import *


def high_boost_filter(gray_image):
    kernel_size = 15

    kernel = np.full((kernel_size, kernel_size), -1, dtype=np.float)

    kernel[7, 7] = 225

    print(gray_image.shape)

    image = np.uint8(gray_image)

    # TODO: 指領域に対してのみヒストグラム平坦化処理を行うように変更する
    cv2.equalizeHist(image, image)
    boosted_image = cv2.filter2D(image, -1, kernel)
    median_image = cv2.medianBlur(boosted_image, 5)

    return median_image

def unet_masking(gray_image):
    size = (gray_image.shape[1], gray_image.shape[0])
    images = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    image = cv2.resize(gray_image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image[:, :, np.newaxis]
    images[0] = normalize_x(image)

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
