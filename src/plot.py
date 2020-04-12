# 学習推移をプロットする関数群
# functions fot plotting
import matplotlib.pyplot as plt


# historyを受け取りloss,accuracyの推移グラフを出力する関数
# get graph of loss and accuracy
def plot_loss_accuracy(history):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))

    # Plot the loss in the history
    axL.plot(history.history['loss'], label="loss for training")
    axL.plot(history.history['val_loss'], label="loss for validation")
    x = history.history['val_loss']
    y = history.history['loss']

    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

    # Plot the accuracy in the history
    axR.plot(history.history['dice_coefficient'], label="dice for training")
    #axR.plot(history.history['val_dice_coefficient'], label="dice for validation")
    #dif = map(lambda x1, x2: x1 - x2, history.history['val_loss'], history.history['loss'])
    #axR.plot(list(dif), label="loss difference", color='black')
    #axR.set_title('Accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='lower right')

    # グラフを保存
    plt.show()
    fig.savefig('./loss_accuracy.png')
    plt.close()


# 回転角度に対するdice係数の推移をプロットする関数
# plot dice coefficient from rotated images
def plot_dice_coefficient(thetas, dices):
    fig = plt.figure()
    plt.plot(thetas, dices, label='original dataset')
    #plt.title('dice coefficinet change')
    plt.xlabel(r'$\theta$')
    plt.ylabel('Similarity')
    plt.show()
    fig.savefig('./dice_change.png')
    plt.close()


# plot_dice_coefficient()の拡張版
# same as plot_dice_coefficient() (compare two data)
def plot_dice_coefficient_compare(thetas1, dices1, label1, thetas2, dices2, label2):
    fig = plt.figure()
    plt.plot(thetas1, dices1, label=label1)
    plt.plot(thetas2, dices2, label=label2)
    plt.xlabel(r'$\theta$')
    plt.ylabel('Similarity')
    plt.legend()
    plt.show()
    fig.savefig('./dice_change.png')
    plt.close()


# plot_dice_coefficient()の交差検証用
# plot_dice_coefficient() for cross validation
def plot_dice_coefficient_cv(dice_lists, label_names):
    fig = plt.figure()
    for i, dice_list in enumerate(dice_lists):
        plt.plot(dice_list, label=label_names[i])

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    fig.savefig('./thesis/accuracy_compare.png')
    plt.close()


# マッチ頻度を比較してプロットする関数
# compare matching histogram
def plot_match_frequency_compare(match_num1, frequency1, label1, match_num2, frequency2, label2):
    fig = plt.figure()
    plt.plot(match_num1, frequency1, label=label1)
    plt.plot(match_num2, frequency2, label=label2)
    plt.ylim(0, 10)
    plt.xlabel('Number of matched keypoints')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    fig.savefig('./match_frequency_compare.png')
    plt.close()


# show image histogram
def show_hist(gray_image):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    hist = cv2.calcHist([gray_image], [0], None, histSize=[256], ranges=[0, 256])
    hist = hist.squeeze(axis=-1)

    fig, ax = plt.subplots()
    ax.fill_between(np.arange(256), hist, color="black")
    ax.set_xticks([0, 255])
    ax.set_xlim([0, 255])
    ax.set_xlabel("Pixel Value")

    plt.show()
    plt.close()
    print(hist)


# FARを比較プロットする関数
# compare FAR
def plot_FAR_compare(threshold1, FAR1, label1, threshold2, FAR2, label2):
    fig = plt.figure()
    plt.plot(threshold1, FAR1, label=label1)
    plt.plot(threshold2, FAR2, label=label2)
    plt.ylim(0, 100)
    plt.xlabel('threshold')
    plt.ylabel('FAR[%]')
    plt.legend()
    plt.show()
    fig.savefig('./FAR_compare.png')
    plt.close()
