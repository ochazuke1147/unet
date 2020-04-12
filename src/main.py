# 実行部
# for training and prediction
from src.unet import *
from src.plot import *
from src.timer import *


# if you want to execute only prediction, comment out 'history = train_unet()' and 'plot_loss_accuracy(history)'.

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
timer = Timer()
history = train_unet()
#timer.time_elapsed()
plot_loss_accuracy(history)
#timer.restart()
predict()
timer.time_elapsed()
#plot_dice_coefficient_compare(thetas1, dice_means1, 'Dataset-B(excluding background)', thetas1, dice_means2, 'Dataset-B')
