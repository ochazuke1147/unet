# 実行部
import cv2
from src.unet import *
from src.plot import plot_loss_accuracy
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print(cv2.__version__)

#history = train_unet()
#plot_loss_accuracy(history)
predict_rotation()
