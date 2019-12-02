from src.segnet import *
from src.plot import *
from src.timer import *


network = segnet(1, 1, FIRST_LAYER_FILTER_COUNT)
timer = Timer()
history = train_segnet()
timer.time_elapsed()
plot_loss_accuracy(history)
timer.reset()
segnet_predict()
timer.time_elapsed()
