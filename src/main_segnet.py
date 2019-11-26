from src.segnet import *
from src.plot import *
from src.timer import *


#network = segnet(1, 1, 64)
#network.summary()
timer = Timer()
#history = train_segnet()
#timer.time_elapsed()
#plot_loss_accuracy(history)

segnet_predict()
timer.time_elapsed()
