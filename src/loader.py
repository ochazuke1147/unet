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
def load_x(folder_path, flip=False):
    import os
    import cv2
    from src.unet import IMAGE_SIZE

    file_names = load_file(folder_path)
    file_names.sort()
    images = np.zeros((len(file_names), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    for i, image_file in enumerate(file_names):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        if flip:
            image = cv2.flip(image, 1)
        image = image[:, :, np.newaxis]
        images[i] = normalize_x(image)
    return images, file_names


# ラベル画像をグレースケールで読み込んで返す関数
def load_y(folder_path):
    import os
    import cv2
    from src.unet import IMAGE_SIZE

    image_files = load_file(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image[:, :, np.newaxis]
        images[i] = normalize_y(image)
    return images
