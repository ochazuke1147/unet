# 実行部
from src.unet import *
from src.plot import *
from src.timer import *


#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
timer = Timer()
#history = train_unet()
#timer.time_elapsed()
#plot_loss_accuracy(history)
#timer.restart()
predict()
timer.time_elapsed()
#thetas2, dice_means2 = predict_rotation()
#thetas1 = range(0, 361, 10)
#dice_means2 = [0.9415533483028412, 0.9259620428085327, 0.9230672478675842, 0.940448260307312, 0.9213405728340149, 0.9155323326587677, 0.9414198696613312, 0.9201304495334626, 0.919729870557785, 0.9410997688770294, 0.9254442572593689, 0.9199683547019959, 0.941605019569397, 0.920464038848877, 0.9232212424278259, 0.942307037115097, 0.919352525472641, 0.9213146328926086, 0.9436696350574494, 0.9198253750801086, 0.9163881957530975, 0.9396203994750977, 0.9236054956912995, 0.9178934752941131, 0.9415533483028412]
#dice_means1 = [0.9418146848678589, 0.9342960000038147, 0.9267257213592529, 0.9278762698173523, 0.9415996432304382, 0.9382441341876984, 0.9285502672195435, 0.92994065284729, 0.9362608850002289, 0.9409989655017853, 0.9337730646133423, 0.9305002510547637, 0.9342273950576783, 0.9434618055820465, 0.9366966784000397, 0.9289986073970795, 0.9315588474273682, 0.9378376960754394, 0.9432358324527741, 0.9357005655765533, 0.9290919423103332, 0.9326129257678986, 0.9407646358013153, 0.9388051092624664, 0.9303401291370392, 0.9291474163532257, 0.9330187618732453, 0.9413659512996674, 0.9372746765613555, 0.9322826027870178, 0.9339288294315338, 0.9425108730792999, 0.9355783581733703, 0.9291255950927735, 0.9294192910194397, 0.9379953980445862, 0.9418146848678589]
#plot_dice_coefficient_compare(thetas1, dice_means1, 'Dataset-B(excluding background)', thetas1, dice_means2, 'Dataset-B')
