# 画像をロードする関数群
import numpy as np
from src.normalize import normalize_x, normalize_y


# 受け取ったパス下のファイル/ディレクトリのうち,'.DS_Store'以外のファイルのlistを返す関数
def load_file(folder_path):
    import os

    file_list = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)) and not filename.startswith('.'):
            file_list.append(filename)
    return file_list


# 受け取ったパス下の静脈画像をグレースケールで読み込み,ファイル名とセットで返す関数
def load_x(folder_path, rotate=False, theta=0):
    import os
    import cv2
    from src.unet import IMAGE_SIZE

    file_names = load_file(folder_path)
    file_names.sort()
    images = np.zeros((len(file_names), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    for i, image_file in enumerate(file_names):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        if rotate:
            image = padding_image(image)
            image = rotate_image(image, theta)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image[:, :, np.newaxis]
        images[i] = normalize_x(image)
    return images, file_names


# ラベル画像をグレースケールで読み込んで返す関数
def load_y(folder_path, rotate=False, theta=0):
    import os
    import cv2
    from src.unet import IMAGE_SIZE

    image_files = load_file(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        if rotate:
            image = padding_image(image)
            image = rotate_image(image, theta)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image[:, :, np.newaxis]
        images[i] = normalize_y(image)
    return images


# 回転前に正方形領域をcropする関数
def crop_image(image, width_start):
    import cv2

    size = image.shape[0]
    cropped = image[0:size, width_start:width_start+size]
    return cropped


# グレースケール画像を回転して返す関数
def rotate_image(image, theta):
    import cv2

    scale = 1.0
    rotation_center = (int(image.shape[0]/2), int(image.shape[1]/2))
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, theta, scale)
    rotated = cv2.warpAffine(image, rotation_matrix, image.shape, flags=cv2.INTER_CUBIC)

    return rotated


# 黒背景を追加して返す関数
def padding_image(image):
    import cv2
    height, width = image.shape[:2]
    size = 1280
    start_height, fin_height = int((size - height)/2), int((size + height)/2)
    start_width, fin_width = int((size - width)/2), int((size + width)/2)

    background = cv2.resize(np.zeros((1, 1, 1), np.uint8), (size, size))
    background[start_height:fin_height, start_width:fin_width] = image

    return background
