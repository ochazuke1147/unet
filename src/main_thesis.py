import cv2
from src.func_processing import opening_masking, segnet_masking


video_paths = ['./datasets/movie/hayashi_n4.avi', './datasets/movie/kikuchi_n2.avi',
               './datasets/movie/kurose_n1.avi', './datasets/movie/okazawa8.avi']

cap = cv2.VideoCapture(video_paths[0])

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_masked = opening_masking(img)
    img_masked = segnet_masking(img)
    cv2.imshow('', img_masked)
    key = cv2.waitKey()
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit()
    elif key == ord('s'):
        cv2.imwrite('thesis/original.png', img[0:700, 0:900])
        cv2.imwrite('thesis/masked.png', img_masked[0:700, 0:900])


