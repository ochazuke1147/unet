from src.segnet import *
from src.plot import *
from src.timer import *


#cross_validation_segnet()
#exit()

network = segnet(1, 1, FIRST_LAYER_FILTER_COUNT)
timer = Timer()
#history = train_segnet()
#plot_loss_accuracy(history)
timer.time_elapsed()
timer.reset()
segnet_predict()
timer.time_elapsed()
