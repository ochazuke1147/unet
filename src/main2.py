import os
from src.func_processing import *
from src.authentication import *
from src.loader import *


registrant_video_path = './datasets/movie/kurose_n1.avi'
user_video_path = './datasets/movie/hayashi_n4.avi'
db = AkazeDB('myname', registrant_video_path)

db.filter_keypoints(3, 10)

#match_numbers = (db.check_matches(user_video_path, 10, 1))
match_numbers = (db.check_matches(registrant_video_path, 10, 10))

print(match_numbers, db.keypoints_DB_number)
#exit()

print(db.check_frequency(db.keypoints_DB_number, match_numbers, 10))

rate = db.check_rate(match_numbers, 0.6)

print(rate)

exit()


cv2.destroyAllWindows()
