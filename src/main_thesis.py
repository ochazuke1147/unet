import cv2
from src.func_processing import opening_masking, segnet_masking

img = cv2.imread('./nohand.png', 0)
ret, bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('binary.png', bin)


video_paths = ['./datasets/movie/hayashi_n4.avi', './datasets/movie/kikuchi_n2.avi',
               './datasets/movie/kurose_n1.avi', './datasets/movie/okazawa8.avi']

cap = cv2.VideoCapture(video_paths[0])

while cap.isOpened():
    ret, img = cap.read()
    img_hist = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(img_hist, img_hist)
    #img_masked = opening_masking(img)
    #img_masked = segnet_masking(img)
    cv2.imshow('', img)
    key = cv2.waitKey()
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit()
    elif key == ord('s'):
        cv2.imwrite('thesis/original.png', img)
        cv2.imwrite('thesis/hist_eq.png', img_hist)


