import cv2
import numpy as np


def high_boost_filter(gray_image):
    kernel_size = 15

    kernel = np.full((kernel_size, kernel_size), -1, dtype=np.float)

    kernel[7, 7] = 225

    print(gray_image.shape)

    gray_image = np.uint8(gray_image)

    #cv2.equalizeHist(gray_image, gray_image)
    cv2.filter2D(gray_image, -1, kernel, gray_image)

    cv2.imshow('', gray_image)
    cv2.waitKey(0)

    print(kernel)
