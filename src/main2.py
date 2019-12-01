import os
from src.func_processing import *
from src.authentication import *
from src.loader import *
from src.plot import *

#FRR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.033333333333333326, 0.033333333333333326, 0.033333333333333326, 0.06666666666666665, 0.09999999999999998, 0.1333333333333333, 0.16666666666666663, 0.16666666666666663, 0.16666666666666663, 0.2666666666666667, 0.30000000000000004, 0.30000000000000004, 0.4666666666666667, 0.5, 0.7, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333]
#FAR = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9666666666666667, 0.9666666666666667, 0.8333333333333334, 0.6666666666666666, 0.5, 0.23333333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#plot_match_frequency_compare(range(100), FRR, 'FRR', range(100), FAR, 'FAR')





registrant_video_path = './datasets/movie/kurose_n1.avi'
user_video_path = './datasets/movie/hayashi_n4.avi'
db = AkazeDB('myname', registrant_video_path)

db.filter_keypoints(3, 10)

match_numbers_others = (db.check_matches(user_video_path, check_number=30, first_frame_number=10))
match_numbers_self = (db.check_matches(registrant_video_path, check_number=30, first_frame_number=10))

db.calc_EER(match_numbers_self, match_numbers_others)

exit()


cv2.destroyAllWindows()
