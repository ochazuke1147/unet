from src.segnet import *

network = segnet(1, 1, 64)
network.summary()
train_segnet()

segnet_predict()