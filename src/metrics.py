# lossの計算関数群
import keras.backend as K


# ダイス係数を計算する関数
def dice_coefficient(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    #print((K.get_value(y_true)))

    dice = 2 * intersection / (K.sum(y_true) + K.sum(y_pred))
    #print(K.get_value(dice))
    return dice
# dice係数の分母に+1してあったが消してみた


# ロス関数
def dice_coefficient_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


# Jaccard係数を計算する関数
def jaccard_coefficient(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)

    print(max(y_pred, y_true))