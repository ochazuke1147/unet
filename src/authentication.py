# implementations for authentication
import cv2
import numpy as np
from src.func_processing import *
from src.plot import *


# akaze特徴点を保持し認証を行うクラス
class AkazeDB:
    def __init__(self, registrant, video_path, first_image_number=0, opening=True):
        self.registrant = registrant
        self.opening = opening
        self.cap = cv2.VideoCapture(video_path)
        #self.cap.set(cv2.CAP_PROP_POS_FRAMES, first_image_number)
        self.image_number = first_image_number
        ret, self.image_DB = self.cap.read()
        if not ret:
            print('image_DB load error!')
            exit(1)
        # preprocess image_DB
        self.image_DB_gray = cv2.cvtColor(self.image_DB, cv2.COLOR_BGR2GRAY)
        print(self.image_DB_gray.dtype)
        if self.opening:
            self.image_DB_mask, self.image_DB_masked = opening_masking(self.image_DB_gray)
        else:
            self.image_DB_mask, self.image_DB_masked = segnet_masking(self.image_DB_gray)
        self.image_DB_processed = highlight_vein(self.image_DB_masked, self.image_DB_mask)
        cv2.imshow('', self.image_DB_processed)
        cv2.waitKey()

        # detect and compute akaze features

        self.akaze = cv2.AKAZE_create()
        self.keypoints_DB, self.descriptors_DB = self.akaze.detectAndCompute(self.image_DB_processed, None)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keypoints_DB_number = len(self.keypoints_DB)
        #print(self.descriptors_DB.shape)

        # list for filtering keypoints
        self.match_numbers = []
        for i in range(len(self.keypoints_DB)):
            self.match_numbers.append(0)

        # definition of threshold
        self.threshold_filter = 70
        self.threshold_check_rate = 0.8

    def show_keypoints(self):
        image = cv2.drawKeypoints(self.image_DB_processed, self.keypoints_DB, None, color=(0, 0, 255), flags=2)
        cv2.imshow('keypoints_DB', image)
        key = cv2.waitKey()

        if key == ord('s'):
            cv2.imwrite('thesis/DB_keypoints.png', image)

    # 特徴点を絞り込むmethod
    def filter_keypoints(self, filter_number, skip_number):
        filter_count = 0
        filtered_keypoints_DB = []
        filtered_descriptors_DB = np.zeros((0, 61), np.uint8)
        while filter_count < filter_number:
            self.image_number += skip_number
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.image_number)
            #print('今：', self.image_number)
            ret, image_filter = self.cap.read()
            if not ret:
                print('image_filter load error!')
            cv2.imwrite('thesis/'+str(filter_count)+'.png', image_filter)
            image_filter_gray = cv2.cvtColor(image_filter, cv2.COLOR_BGR2GRAY)
            if self.opening:
                image_filter_mask, image_filter_masked = opening_masking(image_filter_gray)
            else:
                image_filter_mask, image_filter_masked = segnet_masking(image_filter_gray)
            image_filter_processed = highlight_vein(image_filter_masked, image_filter_mask)
            keypoints_filter, descriptors_filter = self.akaze.detectAndCompute(image_filter_processed, None)

            matches = self.bf_matcher.match(self.descriptors_DB, descriptors_filter)

            matches = sorted(matches, key=lambda x: x.distance)
            matches = self.filter_matches(matches)

            #result = cv2.drawMatches(self.image_DB_masked, self.keypoints_DB, image_filter_masked, keypoints_filter, matches[:], None, flags=2)

            for match in matches:
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

        #self.show_keypoints()

    @staticmethod
    def filter_matches(matches, threshold=90):
        filtered_match = []
        for match in matches:
            if match.distance < threshold:
                filtered_match.append(match)
            #print(match.queryIdx, match.trainIdx)

        return filtered_match

    # DBとのマッチングを行い,各画像とのマッチ数をlistで返すmethod
    def check_matches(self, video_path, check_number, first_frame_number=0, skip_number=2):
        from src.unet import UNet
        from src.segnet import segnet

        input_channel_count = 1
        output_channel_count = 1

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

            if self.opening:
                mask, masked = opening_masking(image_user_gray)
            else:
                mask, masked = segnet_masking(image_user_gray)

            image_user_masked = masked
            image_user_processed = highlight_vein(image_user_masked, mask)
            keypoints_user, descriptors_user = self.akaze.detectAndCompute(image_user_processed, None)


            #print(descriptors_user.shape)

            matches = self.bf_matcher.match(self.descriptors_DB, descriptors_user)

            matches = sorted(matches, key=lambda x: x.distance)
            matches = self.filter_matches(matches)

            #print('マッチ数：', len(matches))
            match_numbers.append(len(matches))

            check_count += 1

        return match_numbers

    # マッチ数の頻度ヒストグラムを出力するmethod
    def check_frequency(self, match_numbers):
        self.frequency = []
        for i in range(self.keypoints_DB_number + 1):
            self.frequency.append(0)

        for match_number in match_numbers:
            if match_number > self.keypoints_DB_number:
                print('match_numbersの値が不正です.')
                exit(1)

            self.frequency[match_number] += 1

        return self.frequency

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
        for threshold_rate in np.linspace(0.5, 1, 100):
            print(threshold_rate)

            FRR = self.calc_FRR(match_numbers_self, threshold_rate)
            FAR = self.calc_FAR(match_numbers_others, threshold_rate)

            list_FRR.append(FRR)
            list_FAR.append(FAR)

        print('FRR', list_FRR)
        print('FAR', list_FAR)

        diff = list(map(lambda x, y: abs(x - y), list_FAR, list_FRR))

        plot_dice_coefficient(list_FAR, list_FRR)

        print(diff)
