import os
from src.func_processing import *
from src.loader import *


X_test, file_names = load_x('datasets' + os.sep + 'test' + os.sep + 'image', False)

high_boost_filter(X_test[0])
