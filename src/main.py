# 実行部
import cv2
from src.unet import *
from src.plot import plot_loss_accuracy
import os

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#history = train_unet()
#plot_loss_accuracy(history)
thetas2, dice_means2 = predict_rotation()
#thetas1 = range(0, 361, 15)
#dice_means2 = [0.9415533483028412, 0.9259620428085327, 0.9230672478675842, 0.940448260307312, 0.9213405728340149, 0.9155323326587677, 0.9414198696613312, 0.9201304495334626, 0.919729870557785, 0.9410997688770294, 0.9254442572593689, 0.9199683547019959, 0.941605019569397, 0.920464038848877, 0.9232212424278259, 0.942307037115097, 0.919352525472641, 0.9213146328926086, 0.9436696350574494, 0.9198253750801086, 0.9163881957530975, 0.9396203994750977, 0.9236054956912995, 0.9178934752941131, 0.9415533483028412]
#dice_means1 = [0.95312460064888, 0.9326848685741425, 0.8981038451194763, 0.8544000387191772, 0.813698661327362, 0.784582132101059, 0.7831295430660248, 0.80829998254776, 0.8378238379955292, 0.8579031467437744, 0.881740003824234, 0.9063358426094055, 0.9136906921863556, 0.9004685044288635, 0.8811506986618042, 0.8471450567245483, 0.8175082862377167, 0.7888797044754028, 0.7733419835567474, 0.7910557389259338, 0.8295813202857971, 0.8588269174098968, 0.9037924647331238, 0.9415551900863648, 0.95312460064888]
#plot_dice_coefficient_compare(thetas1, dice_means1, 'Dataset-A', thetas1, dice_means2, 'Dataset-B')
