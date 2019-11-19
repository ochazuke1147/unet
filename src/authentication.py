# implementations for authentication
import cv2
import numpy as np
from src.func_processing import *


# akaze特徴点を保持しておくクラス
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

        # lists for filtering keypoints
        self.match_numbers = []
        for i in range(len(self.keypoints_DB)):
            self.match_numbers.append(0)

    def show_keypoints(self):
        image = cv2.drawKeypoints(self.image_DB_processed, self.keypoints_DB, None, flags=2)
        cv2.imshow('keypoints_DB', image)
        cv2.waitKey()


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

        self.show_keypoints()



    @staticmethod
    def filter_matches(matches, threshold=70):
        filtered_match = []
        for match in matches:
            if match.distance < threshold:
                filtered_match.append(match)
            #print(match.queryIdx, match.trainIdx)

        return filtered_match

    def check_matches(self, video_path, check_number, first_frame_number=0, skip_number=10):
        cap_user = cv2.VideoCapture(video_path)
        cap_user.set(cv2.CAP_PROP_POS_FRAMES, first_frame_number)

        check_count = 0
        frame_number = first_frame_number
        while check_count < check_number:
            frame_number += skip_number
            cap_user.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, image_user = cap_user.read()
            if not ret:
                print('image_user load error!')
            image_user_gray = cv2.cvtColor(image_user, cv2.COLOR_BGR2GRAY)
            image_user_masked = unet_masking(image_user_gray)
            image_user_processed = high_boost_filter(image_user_masked)
            keypoints_user, descriptors_user = self.akaze.detectAndCompute(image_user_processed, None)

            matches = self.bf_matcher.match(self.descriptors_DB, descriptors_user)

            matches = sorted(matches, key=lambda x: x.distance)
            matches = self.filter_matches(matches)

            print('マッチ数：', len(matches))

            check_count += 1