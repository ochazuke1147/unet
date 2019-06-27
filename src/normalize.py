# 画素値を正規化/非正規化する関数群


# 値を-1から1に正規化する関数
def normalize_x(image):
    image = image/127.5 - 1
    return image


# 値を0から1に正規化する関数
def normalize_y(image):
    image = image/255
    return image


# 値を0から255に戻す関数
def denormalize_y(image):
    image = image*255
    return image
