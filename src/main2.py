import os
from src.func_processing import *
from src.authentication import *
from src.loader import *


registrant_video_path = './datasets/movie/kurose_n1.avi'
user_video_path = './datasets/movie/hayashi_n4.avi'
db = AkazeDB('myname', registrant_video_path)

db.filter_keypoints(3, 10)

#match_numbers = (db.check_matches(user_video_path, 10, 1))
match_numbers = (db.check_matches(registrant_video_path, check_number=50, first_frame_number=10))

db.calc_EER(match_numbers, match_numbers)

exit()


cv2.destroyAllWindows()
