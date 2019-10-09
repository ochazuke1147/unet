# 学習推移をプロットする関数群
import matplotlib.pyplot as plt


# historyを受け取りloss,accuracyの推移グラフを出力する関数
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
    axR.plot(history.history['acc'], label="loss for training")
    axR.plot(history.history['val_acc'], label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='lower right')

    # グラフを保存
    plt.show()
    fig.savefig('./loss_accuracy.png')
    plt.close()


# 回転角度に対するdice係数の推移をプロットする関数
def plot_dice_coefficient(thetas, dices):
    plt.plot(thetas, dices, label='original dataset')
    plt.title('dice coefficient change with rotation')
    plt.xlabel(r'$\theta$')
    plt.ylabel('dice coefficient')
    plt.show()
    plt.savefig('./dice_change.png')
    plt.close()
