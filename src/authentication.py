# implementations for authentication
import cv2
import numpy as np
from src.func_processing import *
from src.plot import *


# akaze特徴点を保持し認証を行うクラス
class AkazeDB:
    def __init__(self, registrant, video_path, first_image_number=0):
        self.registrant = registrant
        self.cap = cv2.VideoCapture(video_path)
        #self.cap.set(cv2.CAP_PROP_POS_FRAMES, first_image_number)
        self.image_number = first_image_number
        ret, self.image_DB = self.cap.read()
        if not ret:
            print('image_DB load error!')
            exit(1)
        # preprocess image_DB
        self.image_DB_gray = cv2.cvtColor(self.image_DB, cv2.COLOR_BGR2GRAY)
        self.image_DB_masked = unet_masking(self.image_DB_gray)
        self.image_DB_processed = high_boost_filter(self.image_DB_masked)

        # detect and compute akaze features
        self.akaze = cv2.AKAZE_create()
        self.keypoints_DB, self.descriptors_DB = self.akaze.detectAndCompute(self.image_DB_processed, None)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keypoints_DB_number = len(self.keypoints_DB)

        # list for filtering keypoints
        self.match_numbers = []
        for i in range(len(self.keypoints_DB)):
            self.match_numbers.append(0)

        # definition of threshold
        self.threshold_filter = 70
        self.threshold_check_rate = 0.8

    def show_keypoints(self):
        image = cv2.drawKeypoints(self.image_DB_processed, self.keypoints_DB, None, flags=2)
        cv2.imshow('keypoints_DB', image)
        cv2.waitKey()

    # 特徴点を絞り込むmethod
    def filter_keypoints(self, filter_number, skip_number):
        filter_count = 0
        filtered_keypoints_DB = []
        filtered_descriptors_DB = np.zeros((0, 61), np.uint8)
        while filter_count < filter_number:
            self.image_number += skip_number
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.image_number)
            print('今：', self.image_number)
            ret, image_filter = self.cap.read()
            if not ret:
                print('image_filter load error!')
            image_filter_gray = cv2.cvtColor(image_filter, cv2.COLOR_BGR2GRAY)
            image_filter_masked = unet_masking(image_filter_gray)
            image_filter_processed = high_boost_filter(image_filter_masked)
            keypoints_filter, descriptors_filter = self.akaze.detectAndCompute(image_filter_processed, None)

            matches = self.bf_matcher.match(self.descriptors_DB, descriptors_filter)

            matches = sorted(matches, key=lambda x: x.distance)
            matches = self.filter_matches(matches)

            result = cv2.drawMatches(self.image_DB_masked, self.keypoints_DB, image_filter_masked, keypoints_filter,
                                     matches[:], None, flags=2)

            for match in matches:
                print(match.queryIdx)
                self.match_numbers[match.queryIdx] += 1

            #cv2.imshow('', result)
            #cv2.waitKey()
            filter_count += 1

        for i, num in enumerate(self.match_numbers):
            if num == filter_number:
                filtered_keypoints_DB.append(self.keypoints_DB[i])
                filtered_descriptors_DB = np.vstack((filtered_descriptors_DB, self.descriptors_DB[i, :]))

        self.keypoints_DB = filtered_keypoints_DB
        self.descriptors_DB = filtered_descriptors_DB
        self.keypoints_DB_number = len(self.keypoints_DB)

        # self.show_keypoints()

    @staticmethod
    def filter_matches(matches, threshold=70):
        filtered_match = []
        for match in matches:
            if match.distance < threshold:
                filtered_match.append(match)
            #print(match.queryIdx, match.trainIdx)

        return filtered_match

    # DBとのマッチングを行い,各画像とのマッチ数をlistで返すmethod
    def check_matches(self, video_path, check_number, first_frame_number=0, skip_number=2):
        input_channel_count = 1
        output_channel_count = 1
        first_layer_filter_count = 64
        network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
        model = network.get_model()
        model.load_weights('unet_weights.hdf5')
        BATCH_SIZE = 1

        cap_user = cv2.VideoCapture(video_path)
        cap_user.set(cv2.CAP_PROP_POS_FRAMES, first_frame_number)
        check_count = 0
        frame_number = first_frame_number
        match_numbers = []
        while check_count < check_number:
            frame_number += skip_number
            cap_user.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, image_user = cap_user.read()
            if not ret:
                print('image_user load error!')
            image_user_gray = cv2.cvtColor(image_user, cv2.COLOR_BGR2GRAY)

            size = (image_user_gray.shape[1], image_user_gray.shape[0])
            images = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
            image = cv2.resize(image_user_gray, (IMAGE_SIZE, IMAGE_SIZE))
            image = image[:, :, np.newaxis]
            images[0] = normalize_x(image)
            Y_pred = model.predict(images, BATCH_SIZE)
            y = cv2.resize(Y_pred[0], size)
            y_dn = denormalize_y(y)
            y_dn = np.uint8(y_dn)
            ret, mask = cv2.threshold(y_dn, 0, 255, cv2.THRESH_OTSU)
            masked = cv2.bitwise_and(image_user_gray, mask)
            mask_rest = cv2.bitwise_not(mask)
            masked = cv2.bitwise_or(masked, mask_rest)


            image_user_masked = masked
            image_user_processed = high_boost_filter(image_user_masked)
            keypoints_user, descriptors_user = self.akaze.detectAndCompute(image_user_processed, None)

            matches = self.bf_matcher.match(self.descriptors_DB, descriptors_user)

            matches = sorted(matches, key=lambda x: x.distance)
            matches = self.filter_matches(matches)

            print('マッチ数：', len(matches))
            match_numbers.append(len(matches))

            check_count += 1

        return match_numbers

    # マッチ数の頻度ヒストグラムを出力するmethod
    def check_frequency(self, keypoints_DB_number, match_numbers, check_number):
        frequency = []
        for i in range(keypoints_DB_number+1):
            frequency.append(0)

        for match_number in match_numbers:
            if match_number > keypoints_DB_number:
                print('match_numbersの値が不正です.')
                exit(1)

            frequency[match_number] += 1

        return frequency

    # match_numbersを受け取り,threshold以上のマッチ数ならaccept,未満ならrejectとして扱い,受容率を返すmethod
    def check_rate(self, match_numbers, threshold_rate):
        accept_number = 0
        for match_number in match_numbers:
            if match_number >= (self.keypoints_DB_number * threshold_rate):
                accept_number += 1

        accept_rate = accept_number/len(match_numbers)
        return accept_rate

    # FRRを計算するmethod
    def calc_FRR(self, match_numbers, threshold_rate):
        FRR = 1 - self.check_rate(match_numbers, threshold_rate)
        return FRR

    # FARを計算するmethod
    def calc_FAR(self, match_numbers, threshold_rate):
        FAR = self.check_rate(match_numbers, threshold_rate)
        return FAR

    # EERを計算するmethod
    def calc_EER(self, match_numbers_self, match_numbers_others):
        list_FRR = []
        list_FAR = []
        for threshold_rate in np.linspace(0.0, 1, 100):
            print(threshold_rate)

            FRR = self.calc_FRR(match_numbers_self, threshold_rate)
            FAR = self.calc_FAR(match_numbers_others, threshold_rate)

            list_FRR.append(FRR)
            list_FAR.append(FAR)

        print('FRR', list_FRR)
        print('FAR', list_FAR)

        diff = list(map(lambda x, y: abs(x - y), list_FAR, list_FRR))

        plot_match_frequency_compare(range(100), list_FRR, 'FRR', range(100), list_FAR, 'FAR')

        print(diff)
