# U-Netを操作するクラス,関数群
from keras.models import Model
from keras.layers import Input, LeakyReLU, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.optimizers import Adam
import os
from src.loader import load_x, load_y
from src.metrics import dice_coefficient, dice_coefficient_loss
from src.normalize import denormalize_y

# imageは(256, 256, 1)で読み込み
IMAGE_SIZE = 256
# 一番初めのConvolutionフィルタ枚数は32
FIRST_LAYER_FILTER_COUNT = 64

# convolution後のshape_size=(size_old-filter_size)/stride+1


# U-Netのネットワークを構築するクラス
class UNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        # 以下,first_layer_filter_count:Nと表記
        self.INPUT_IMAGE_SIZE = IMAGE_SIZE
        self.CONCATENATE_AXIS = -1
        # チャンネルの軸で結合することを指定している
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        # build NN with functional API
        input_img = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        # (256, 256, input_channel_count)

        enc1 = ZeroPadding2D(self.CONV_PADDING)(input_img)
        # (258, 258, input_channel_count)

        enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)
        # (128, 128, N)

        filter_count = first_layer_filter_count*2
        enc2 = self._add_encoding_layer(filter_count, enc1)
        # (64, 64, 2N)

        filter_count = first_layer_filter_count*4
        enc3 = self._add_encoding_layer(filter_count, enc2)
        # (32, 32, 4N)

        filter_count = first_layer_filter_count*8
        enc4 = self._add_encoding_layer(filter_count, enc3)
        # (16, 16, 8N)

        enc5 = self._add_encoding_layer(filter_count, enc4)
        # (8, 8, 8N)

        dec4 = self._add_decoding_layer(filter_count, True, enc5)
        # (16, 16, 8N)

        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)
        # (16, 16, 16N)

        filter_count = first_layer_filter_count*4
        dec5 = self._add_decoding_layer(filter_count, True, dec4)
        # (32, 32, 4N)

        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)
        # (32, 32, 8N)

        filter_count = first_layer_filter_count*2
        dec6 = self._add_decoding_layer(filter_count, True, dec5)
        # (64, 64, 2N)

        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)
        # (64, 64, 4N)

        filter_count = first_layer_filter_count
        dec7 = self._add_decoding_layer(filter_count, True, dec6)
        # (128, 128, N)

        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)
        # (128, 128 ,2N)

        dec8 = Activation(activation='relu')(dec7)
        dec8 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        # (256, 256, output_channel_count)

        dec8 = Activation(activation='sigmoid')(dec8)

        self.UNet = Model(inputs=input_img, outputs=dec8)

        # self.UNet.summary()

    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self):
        return self.UNet


# U-Netをtrainingする関数
def train_unet():
    # 訓練用imageデータ読み込み
    x_train, file_names = load_x('datasets' + os.sep + 'training' + os.sep + 'image')
    # 訓練用labelデータ読み込み
    y_train = load_y('datasets' + os.sep + 'training' + os.sep + 'label')
    # 検証用imageデータ読み込み
    x_validation, file_names2 = load_x('datasets' + os.sep + 'test' + os.sep + 'image')
    # 検証用labelデータ読み込み
    y_validation = load_y('datasets' + os.sep + 'test' + os.sep + 'label')

    # 入力はグレースケール1チャンネル
    input_channel_count = 1
    # 出力はグレースケール1チャンネル
    output_channel_count = 1

    # U-Netの生成
    network = UNet(input_channel_count, output_channel_count, FIRST_LAYER_FILTER_COUNT)
    model = network.get_model()
    model.compile(loss=dice_coefficient_loss, optimizer=Adam(lr=1e-4), metrics=[dice_coefficient, 'accuracy'])

    BATCH_SIZE = 5
    # 20エポック回せば十分
    NUM_EPOCH = 500
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1,
                        validation_data=(x_validation, y_validation))
    model.save_weights('unet_weights.hdf5')

    return history


# 学習後のU-Netによる予測を行う関数
def predict():
    import cv2

    # test内の画像で予測
    X_test, file_names = load_x('datasets' + os.sep + 'test' + os.sep + 'image', True)
    # X_test, file_names = load_X('testData' + os.sep + 'left_images')

    input_channel_count = 1
    output_channel_count = 1
    first_layer_filter_count = 64
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.load_weights('unet_weights.hdf5')
    BATCH_SIZE = 12
    Y_pred = model.predict(X_test, BATCH_SIZE)

    for i, y in enumerate(Y_pred):
        # testDataフォルダ配下にleft_imagesフォルダを置いている
        img = cv2.imread('datasets' + os.sep + 'test' + os.sep + 'image' + os.sep + file_names[i], 0)
        # img = cv2.imread('testData' + os.sep + 'left_images' + os.sep + file_names[i])

        y = cv2.resize(y, (img.shape[0], img.shape[1]))
        y_dn = denormalize_y(y)

        cv2.imwrite('prediction' + os.sep + file_names[i], y_dn)
        # img_pre = cv2.imread('prediction' + str(i) + '.png')
        # img_gt = cv2.imread('testData' + os.sep + 'left_groundTruth' + os.sep + file_names[i])
        # img_compare = cv2.hconcat([img_pre, img_gt])
        # cv2.imwrite('compare' + str(i) + '.png', img_compare)
