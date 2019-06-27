import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 1


def createModel():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model

# Mnistデータのロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784) # 2次元配列を1次元に変換(訓練データ)
x_test = x_test.reshape(10000, 784) # 2次元配列を1次元に変換(テストデータ)
x_train = x_train.astype('float32') # int型をfloat32型に変換
x_test = x_test.astype('float32') # int型をfloat32型に変換
x_train /= 255                     # [0-255]の値を[0.0-1.0]に変換
x_test /= 255

# 正解ラベルのOne hot vector化
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# モデルの定義
model = createModel()

model.summary()

# 学習の実行
history = model.fit(x_train, y_train,  # 画像とラベルデータ
                    batch_size=batch_size,
                    epochs=epochs,     # エポック数の指定
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])