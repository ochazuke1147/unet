import os
from src.func_processing import *
from src.loader import *


cap = cv2.VideoCapture('./datasets/movie/okazawa8.avi')

print(cap.isOpened())

window_name = 'image'

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        unet_masking(frame)
        #frame = high_boost_filter(frame)
        #cv2.imshow(window_name, frame)
        if cv2.waitKey(0) == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.destroyWindow(window_name)