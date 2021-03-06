# this code defines segnet(encoder-decoder) model architecture

import os
if os.name == 'posix':
    print('on macOS')
    #import plaidml.keras
    #plaidml.keras.install_backend()
from keras.layers import Input, Dropout
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
import keras.backend as K

from src.loader import *
from src.metrics import dice_coefficient, dice_coefficient_loss, jaccard_coefficient
from src.normalize import denormalize_y
from src.func_processing import *

# imageは(128, 128, 1)で読み込み
# load image as (128, 128, 1) size (128x128, grayscale)
IMAGE_SIZE = 128
# 一番初めのConvolutionフィルタ枚数は64
# set model parameter
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
    # load images for training
    x_train, file_names = load_x('datasets' + os.sep + 'Dataset-A' + os.sep + 'training' + os.sep + 'image')
    # 訓練用labelデータ読み込み
    # load label images for training
    y_train = load_y('datasets' + os.sep + 'Dataset-A' + os.sep + 'training' + os.sep + 'label')
    # 検証用imageデータ読み込み
    # load images for validation
    x_validation, file_names2 = load_x('datasets' + os.sep + 'validation' + os.sep + 'image')
    # 検証用labelデータ読み込み
    # load label images for validation
    y_validation = load_y('datasets' + os.sep + 'validation' + os.sep + 'label')

    # 入力はグレースケール1チャンネル
    # input: grayscale 1ch
    input_channel_count = 1
    # 出力はグレースケール1チャンネル
    # output: grayscale 1ch
    output_channel_count = 1

    # SegNetの定義
    # build segnet model
    model = segnet(input_channel_count, output_channel_count, FIRST_LAYER_FILTER_COUNT)

    model.compile(loss=dice_coefficient_loss, optimizer=Adam(lr=1e-3), metrics=[dice_coefficient, 'accuracy'])

    # batch size
    BATCH_SIZE = 8
    # epoch number
    NUM_EPOCH = 30

    # get training history
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1,
                        validation_data=(x_validation, y_validation))
    model.save_weights('segnet_weights.hdf5')

    return history


# prediction using SegNet
def segnet_predict():
    import cv2
    from src.timer import Timer

    rotation = False
    # set True for running segmentation test
    segmentation_test = True

    jaccard_sum_previous = 0
    jaccard_sum_proposed = 0

    time_previous = [0, 0, 0, 0]
    time_proposed = 0

    if segmentation_test:
        # 指領域抽出実験用
        # for segmentation test
        X_test, file_names = load_x('datasets' + os.sep + 'segmentation_test' + os.sep + 'image', rotation)
    else:
        # normal predict
        X_test, file_names = load_x('datasets' + os.sep + 'test' + os.sep + 'image', rotation)

    input_channel_count = 1
    output_channel_count = 1
    first_layer_filter_count = FIRST_LAYER_FILTER_COUNT
    model = segnet(input_channel_count, output_channel_count, first_layer_filter_count)
    model.load_weights('segnet_weights.hdf5')
    model.summary()
    BATCH_SIZE = 8
    timer = Timer()
    Y_pred = model.predict(X_test, BATCH_SIZE)
    #time_proposed += timer.time_elapsed()

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

        timer.reset()
        masked = cv2.bitwise_and(img, mask)
        mask_rest = cv2.bitwise_not(mask)
        masked = cv2.bitwise_or(masked, mask_rest)
        #image_user_processed = high_boost_filter(masked)
        cv2.imwrite('prediction' + os.sep + file_names[i], mask)
        time_proposed += timer.time_elapsed()

        if segmentation_test:
            img = cv2.imread('datasets' + os.sep + 'segmentation_test' + os.sep + 'image' + os.sep + file_names[i], 0)
            mask_previous, _, time_list_previous = opening_masking(img)
            time_previous[0] += time_list_previous[0]
            time_previous[1] += time_list_previous[1] - time_list_previous[0]
            time_previous[2] += time_list_previous[2] - time_list_previous[1]
            time_previous[3] += time_list_previous[3] - time_list_previous[2]

            mask_previous_binary = normalize_y(mask_previous)
            print(mask_previous_binary.shape)
            label = cv2.imread('datasets' + os.sep + 'segmentation_test' + os.sep + 'label' + os.sep + file_names[i], 0)
            label_binary = normalize_y(label)

            jaccard_previous = K.get_value(jaccard_coefficient(mask_previous_binary, label_binary))
            jaccard_proposed = K.get_value(jaccard_coefficient(mask_binary, label_binary))

            print('previous:', jaccard_previous)
            jaccard_sum_previous += jaccard_previous
            print('proposed:', jaccard_proposed)
            jaccard_sum_proposed += jaccard_proposed

    print('sum_previous:', jaccard_sum_previous)
    print('sum_proposed:', jaccard_sum_proposed)
    print('mean_previous:', jaccard_sum_previous/len(Y_pred))
    print('mean_proposed:', jaccard_sum_proposed/len(Y_pred))
    print('mean_time_previous:', list(map(lambda x: x / len(Y_pred), time_previous)))
    print('mean_time_proposed', time_proposed/len(Y_pred))

    return 0


# 4分割交差検証を行う関数
# function for 4-fold cross validation
def cross_validation_segnet():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from src.plot import plot_loss_accuracy, plot_dice_coefficient_cv
    from src.timer import Timer

    segmentation_test = True
    rotation = False
    jaccards_previous = []
    jaccards_proposed = []

    # training_validation dataset path
    training_validation_path = 'datasets' + os.sep + 'training_validation'

    # 学習・検証用データセットのロード
    # load dataset for training, validation
    x_all, file_names = load_x(training_validation_path + os.sep + 'image')
    y_all = load_y(training_validation_path + os.sep + 'label')
    # testデータの分割だがこれは今回あらかじめ分けておくので無視する
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=7)
    # 指領域抽出実験用
    X_test, file_names = load_x('datasets' + os.sep + 'segmentation_test' + os.sep + 'image', rotation)

    # 入力はグレースケール1チャンネル
    input_channel_count = 1
    # 出力はグレースケール1チャンネル
    output_channel_count = 1
    # hyper parameters
    BATCH_SIZE = 8
    NUM_EPOCH = 300

    # dice係数の最終値を記憶するlist
    # lists for get last dice coefficient values
    final_dices = []
    final_val_dices = []
    # dice係数のlistを記憶するlist
    # list for get dice coefficient
    dice_lists = []

    training_times = []

    label_names = ['1st', '2nd', '3rd', '4th']

    kf = KFold(n_splits=4, shuffle=True)
    timer = Timer()
    # kFoldループを行う(36データを4つに分割)
    # kFold loop (split 36 data to 4 data)
    for train_index, val_index in kf.split(x_all, y_all):
        print(train_index, val_index)
        x_train = x_all[train_index]
        y_train = y_all[train_index]
        x_validation = x_all[val_index]
        y_validation = y_all[val_index]

        print(x_train.shape, x_validation.shape)

        # SegNetの定義
        model = segnet(input_channel_count, output_channel_count, FIRST_LAYER_FILTER_COUNT)
        model.compile(loss=dice_coefficient_loss, optimizer=Adam(lr=1e-3), metrics=[dice_coefficient, 'accuracy'])
        timer.reset()
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1,
                            validation_data=(x_validation, y_validation))
        training_times.append(timer.time_elapsed())

        # dice_coefficientの最終値を記録
        # get last dice coefficient value
        final_dices.append(history.history['dice_coefficient'][-1])
        final_val_dices.append(history.history['val_dice_coefficient'][-1])
        dice_lists.append(history.history['dice_coefficient'])
        plot_loss_accuracy(history)
        #Y_pred = model.predict(X_test, BATCH_SIZE)

        if segmentation_test:
            Y_pred = model.predict(X_test, BATCH_SIZE)

            for i, y in enumerate(Y_pred):
                # testDataフォルダ配下にleft_imagesフォルダを置いている
                img = cv2.imread('datasets' + os.sep + 'segmentation_test' + os.sep + 'image' + os.sep + file_names[i], 0)
                if rotation:
                    y = cv2.resize(y, (img.shape[0], img.shape[0]))
                else:
                    y = cv2.resize(y, (img.shape[1], img.shape[0]))

                # 提案手法
                # proposed method
                y_dn = denormalize_y(y)
                y_dn = np.uint8(y_dn)
                # ret, mask = cv2.threshold(y_dn, 0, 255, cv2.THRESH_OTSU)
                ret, mask = cv2.threshold(y_dn, 127, 255, cv2.THRESH_BINARY)
                # hist, bins = np.histogram(mask.ravel(), 256, [0, 256])
                mask_binary = normalize_y(mask)
                # 従来手法
                # previous method
                mask_previous, _ = opening_masking(img)
                mask_previous_binary = normalize_y(mask_previous)
                # ラベル読み込み
                # load label images
                label = cv2.imread(
                    'datasets' + os.sep + 'segmentation_test' + os.sep + 'label' + os.sep + file_names[i], 0)
                label_binary = normalize_y(label)

                dice_previous = K.get_value(jaccard_coefficient(mask_previous_binary, label_binary))
                dice_proposed = K.get_value(jaccard_coefficient(mask_binary, label_binary))

                print('previous:', dice_previous)
                jaccards_previous.append(dice_previous)
                print('proposed:', dice_proposed)
                jaccards_proposed.append(dice_proposed)

    print(final_dices)
    print('平均訓練精度', np.mean(final_dices))  # mean training precision
    print(final_val_dices)
    print('平均検証精度', np.mean(final_val_dices))  # mean validation precision
    print(training_times)
    print('平均学習時間', np.mean(training_times))  # mean training time
    print(jaccards_previous)
    print('従来手法精度', np.mean(jaccards_previous))  # extract-precision by previous method
    print(jaccards_proposed)
    print('提案手法精度', np.mean(jaccards_proposed))  # extract-precision by proposed method

    plot_dice_coefficient_cv(dice_lists, label_names)

    return 0
