import os
from src.func_processing import *
from src.authentication import *
from src.loader import *


registrant_video_path = './datasets/movie/kurose_n1.avi'
user_video_path = './datasets/movie/hayashi_n4.avi'
db = AkazeDB('myname', registrant_video_path)

db.filter_keypoints(3, 10)

db.check_matches(user_video_path, 2, 1)

exit()


cv2.destroyAllWindows()
