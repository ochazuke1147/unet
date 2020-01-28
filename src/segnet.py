import os
if os.name == 'posix':
    print('on macOS')
    import plaidml.keras
    plaidml.keras.install_backend()
from keras.layers import Input, Dropout
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
import keras.backend as K

from src.loader import *
from src.metrics import dice_coefficient, dice_coefficient_loss
from src.normalize import denormalize_y
from src.func_processing import *

# imageは(256, 256, 1)で読み込み
IMAGE_SIZE = 128
# 一番初めのConvolutionフィルタ枚数は64
FIRST_LAYER_FILTER_COUNT = 64


# SegNet model definition
def segnet(input_channel_count, output_channel_count, first_layer_filter_count):
    INPUT_IMAGE_SIZE = IMAGE_SIZE

    # build NN with functional API
    input_img = Input((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, input_channel_count))
    # (256, 256, input_channel_count)

    x = input_img

    # Encoder
    x = Conv2D(first_layer_filter_count, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(first_layer_filter_count*2, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(first_layer_filter_count*4, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(first_layer_filter_count*8, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Decoder
    x = Conv2D(first_layer_filter_count*8, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(first_layer_filter_count*4, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(first_layer_filter_count*2, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(first_layer_filter_count, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Activation("relu")(x)

    x = Conv2D(output_channel_count, (1, 1), padding="valid")(x)

    x = Activation(activation='sigmoid')(x)
    model = Model(inputs=input_img, outputs=x)

    return model


# train SegNet model
def train_segnet():
    # training/validation dataset paths
    training_path = 'datasets' + os.sep + 'training'
    validation_path = 'datasets' + os.sep + 'validation'

    # 訓練用imageデータ読み込み
    x_train, file_names = load_x('datasets' + os.sep + 'training' + os.sep + 'image')
    # 訓練用labelデータ読み込み
    y_train = load_y('datasets' + os.sep + 'training' + os.sep + 'label')
    # 検証用imageデータ読み込み
    x_validation, file_names2 = load_x('datasets' + os.sep + 'validation' + os.sep + 'image')
    # 検証用labelデータ読み込み
    y_validation = load_y('datasets' + os.sep + 'validation' + os.sep + 'label')

    # 入力はグレースケール1チャンネル
    input_channel_count = 1
    # 出力はグレースケール1チャンネル
    output_channel_count = 1

    # SegNetの定義
    model = segnet(input_channel_count, output_channel_count, FIRST_LAYER_FILTER_COUNT)

    model.compile(loss=dice_coefficient_loss, optimizer=Adam(lr=1e-3), metrics=[dice_coefficient, 'accuracy'])

    BATCH_SIZE = 8
    # 20エポック回せば十分
    NUM_EPOCH = 300

    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1,
                        validation_data=(x_validation, y_validation))
    model.save_weights('segnet_weights.hdf5')

    return history


# prediction using SegNet
def segnet_predict():
    import cv2

    rotation = False
    segmentation_test = False

    dice_sum_previous = 0
    dice_sum_proposed = 0

    if segmentation_test:
        # 指領域抽出実験用
        X_test, file_names = load_x('datasets' + os.sep + 'segmentation_test' + os.sep + 'image', rotation)
    else:
        # 普通のpredict
        X_test, file_names = load_x('datasets' + os.sep + 'test' + os.sep + 'image', rotation)

    input_channel_count = 1
    output_channel_count = 1
    first_layer_filter_count = FIRST_LAYER_FILTER_COUNT
    model = segnet(input_channel_count, output_channel_count, first_layer_filter_count)
    model.load_weights('segnet_weights.hdf5')
    model.summary()
    BATCH_SIZE = 12
    Y_pred = model.predict(X_test, BATCH_SIZE)

    for i, y in enumerate(Y_pred):
        # testDataフォルダ配下にleft_imagesフォルダを置いている
        img = cv2.imread('datasets' + os.sep + 'test' + os.sep + 'image' + os.sep + file_names[i], 0)
        #cv2.equalizeHist(img ,img)

        if rotation:
            y = cv2.resize(y, (img.shape[0], img.shape[0]))
        else:
            y = cv2.resize(y, (img.shape[1], img.shape[0]))

        y_dn = denormalize_y(y)
        y_dn = np.uint8(y_dn)
        #ret, mask = cv2.threshold(y_dn, 0, 255, cv2.THRESH_OTSU)
        ret, mask = cv2.threshold(y_dn, 127, 255, cv2.THRESH_BINARY)
        #hist, bins = np.histogram(mask.ravel(), 256, [0, 256])
        mask_binary = normalize_y(mask)

        masked = cv2.bitwise_and(img, mask)
        mask_rest = cv2.bitwise_not(mask)
        masked = cv2.bitwise_or(masked, mask_rest)
        image_user_processed = high_boost_filter(masked)
        cv2.imwrite('prediction' + os.sep + file_names[i], image_user_processed)

        if segmentation_test:
            mask_previous = opening_masking(img)
            mask_previous_binary = normalize_y(mask_previous)
            label = cv2.imread('datasets' + os.sep + 'segmentation_test' + os.sep + 'label' + os.sep + file_names[i], 0)
            label_binary = normalize_y(label)

            dice_previous = K.get_value(dice_coefficient(mask_previous_binary, label_binary))
            dice_proposed = K.get_value(dice_coefficient(mask_binary, label_binary))

            print('previous:', dice_previous)
            dice_sum_previous += dice_previous
            print('proposed:', dice_proposed)
            dice_sum_proposed += dice_proposed

    print('dice_sum_previous:', dice_sum_previous)
    print('dice_sum_proposed:', dice_sum_proposed)

    return 0
