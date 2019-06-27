# 実行部
import cv2
from src.unet import train_unet, predict
from src.plot import plot_loss_accuracy

print(cv2.__version__)

history = train_unet()
plot_loss_accuracy(history)
predict()
