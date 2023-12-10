import numpy as np
import imutils
import cv2
import time

class Panaroma:

    def pad_image(image, padding):
        # 获取图像的尺寸
        height, width = image.shape[:2]

        # 创建一个新的图像，尺寸增加 padding 像素的宽度
        padded_image = np.zeros((height, width + padding, image.shape[2] + padding), dtype=np.uint8)

        # 将原始图像复制到新的图像中
        padded_image[:, :width, :] = image
        
        return padded_image

    def get_video_H_and_best_seam(self, images, mask, lowe_ratio=0.5, max_Threshold=4.0):
        (imageB, imageA) = images
        (KeypointsA, features_of_A) = self.Detect_Feature_And_KeyPoints(imageA,mask)
        (KeypointsB, features_of_B) = self.Detect_Feature_And_KeyPoints(imageB,mask)

        #got the valid matched points
        Values = self.matchKeypoints(KeypointsA, KeypointsB,features_of_A, features_of_B, lowe_ratio, max_Threshold)

        if Values is None:
            return None

        #to get perspective of image using computed homography
        (matches, Homography, status) = Values
        result_image = self.getwarp_perspective(imageA,imageB,Homography)
        
        h1, w1 = imageA.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts1, Homography)
        points=[pts2[0][0],pts2[1][0]]
        leftmost = min(points, key=lambda p: p[0])
        imageB1=imageB[:,int(leftmost[0]):,:]
        imageA1=result_image[:,int(leftmost[0]):imageB.shape[1],:]

        C = self.optimal_seam_rule2(imageB1, imageA1)
        C=C+int(leftmost[0])

        return Homography, C

    def get_pad_image(self,image,shape):

        # 将原始图像复制到新的图像中
        image_pad = np.zeros((shape[0],shape[1],3), dtype=np.uint8)
        image_pad[120:600, :, :] = image
        mask = np.zeros((shape[0],shape[1],1), dtype=np.uint8)
        mask[120:600, :, :] = 1
        return image_pad, mask


    def image_stitch_for_video(self, images, H, C, lowe_ratio=0.5, max_Threshold=4.0, match_status=False):
        #detect the features and keypoints from SIFT
        (imageB, imageA) = images

        result_image = self.getwarp_perspective(imageA,imageB,H)
        for i in range(len(C)):
            result_image[i, 0:C[i]] = imageB[i, 0:C[i]]
        # #绘制接缝线
        # for i in range(len(C) - 1):
        #     cv2.line(result_image, (C[i], i), (C[i + 1], i + 1), (0, 255, 0), 1)

        return result_image



    def image_stitch(self, images, lowe_ratio=0.5, max_Threshold=4.0,match_status=False):

        #detect the features and keypoints from SIFT
        (imageB, imageA) = images
        start = time.time()
        (KeypointsA, features_of_A) = self.Detect_Feature_And_KeyPoints(imageA)
        (KeypointsB, features_of_B) = self.Detect_Feature_And_KeyPoints(imageB)

        #got the valid matched points
        Values = self.matchKeypoints(KeypointsA, KeypointsB,features_of_A, features_of_B, lowe_ratio, max_Threshold)
        end = time.time()
        print("match points:{}".format(end - start))

        if Values is None:
            return None

        #to get perspective of image using computed homography
        (matches, Homography, status) = Values
        result_image = self.getwarp_perspective(imageA,imageB,Homography)
        

        h1, w1 = imageA.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts1, Homography)
        points=[pts2[0][0],pts2[1][0]]
        leftmost = min(points, key=lambda p: p[0])
        imageB1=imageB[:,int(leftmost[0]):,:]
        imageA1=result_image[:,int(leftmost[0]):imageB.shape[1],:]

        start = time.time()
        C = self.optimal_seam_rule2(imageB1, imageA1)
        C=C+int(leftmost[0])
        end = time.time()
        print("optimal seam={}".format(end - start))
        for i in range(len(C)):
            result_image[i, 0:C[i]] = imageB[i, 0:C[i]]
        #绘制接缝线
        for i in range(len(C) - 1):
            cv2.line(result_image, (C[i], i), (C[i + 1], i + 1), (0, 255, 0), 1)
        #去除多余的黑边
        # result_image=self.capture_image(result_image)

        # result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        start = time.time()
        if match_status:
            vis = self.draw_Matches(imageA, imageB, KeypointsA, KeypointsB, matches,status)
            end = time.time()
            print(end - start)
            return (result_image, vis)

        return result_image

    def optimal_seam_rule_value(self, I1, I2):
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        Sx = np.array([[-2, 0, 2], [-1, 0, 1], [-2, 0, 2]])
        Sy = np.array([[-2, -1, -2], [0, 0, 0], [2, 1, 2]])
    
        I1_Sx = cv2.filter2D(I1, -1, Sx)
        I1_Sy = cv2.filter2D(I1, -1, Sy)
        I2_Sx = cv2.filter2D(I2, -1, Sx)
        I2_Sy = cv2.filter2D(I2, -1, Sy)
    
        E_color = (I1 - I2) ** 2
        E_geometry = (I1_Sx - I2_Sx) * (I1_Sy - I2_Sy)
        E = E_color + E_geometry
        return E.astype(float)
    
    def optimal_seam_rule2(self, I1, I2):
        E = self.optimal_seam_rule_value(I1, I2)
        # optimal seam
        paths_weight = E[0, 1:-1].reshape(1, -1)  # Cumulative strength value
        paths = np.arange(1, E.shape[1] - 1).reshape(1, -1)  # save index
        for i in range(1, E.shape[0]):
            # boundary process
            lefts_index = paths[-1, :] - 1
            lefts_index[lefts_index < 0] = 0
            rights_index = paths[-1, :] + 1
            rights_index[rights_index > E.shape[1] - 1] = E.shape[1] - 1
            mids_index = paths[-1, :]
            mids_index[mids_index < 0] = 0
            mids_index[mids_index > E.shape[1] - 1] = E.shape[1] - 1
    
            # compute next row strength value(remove begin and end point)
            lefts = E[i, lefts_index] + paths_weight[-1, :]
            mids = E[i, paths[-1, :]] + paths_weight[-1, :]
            rights = E[i, rights_index] + paths_weight[-1, :]
            # return the index of min strength value 
            values_3direct = np.vstack((lefts, mids, rights))
            index_args = np.argmin(values_3direct, axis=0) - 1  # 
            # next min strength value and index
            weights = np.min(values_3direct, axis=0)
            path_row = paths[-1, :] + index_args
            paths_weight = np.insert(paths_weight, paths_weight.shape[0], values=weights, axis=0)
            paths = np.insert(paths, paths.shape[0], values=path_row, axis=0)
    
        # search min path
        min_index = np.argmin(paths_weight[-1, :])
        return paths[:, min_index]


    def getwarp_perspective_with_size(self,imageA,imageB,Homography, shape : list):
        
        result_image = cv2.warpPerspective(imageA, Homography, (shape[1], shape[0]))

        return result_image


    def getwarp_perspective(self,imageA,imageB,Homography):
        val = imageA.shape[1] + imageB.shape[1]
        result_image = cv2.warpPerspective(imageA, Homography, (val , imageA.shape[0]))

        return result_image

    def Detect_Feature_And_KeyPoints(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptors = cv2.SIFT_create()
        (Keypoints, features) = descriptors.detectAndCompute(image, mask)

        Keypoints = np.float32([i.pt for i in Keypoints])
        return (Keypoints, features)

    def get_Allpossible_Match(self,featuresA,featuresB):

        # compute the all matches using euclidean distance and opencv provide
        #DescriptorMatcher_create() function for that
        match_instance = cv2.DescriptorMatcher_create("BruteForce")
        All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)

        return All_Matches

    def All_validmatches(self,AllMatches,lowe_ratio):
        #to get all valid matches according to lowe concept..
        valid_matches = []

        for val in AllMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                valid_matches.append((val[0].trainIdx, val[0].queryIdx))

        return valid_matches

    def Compute_Homography(self,pointsA,pointsB,max_Threshold):
        #to compute homography using points in both images

        (H, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
        return (H,status)

    def matchKeypoints(self, KeypointsA, KeypointsB, featuresA, featuresB,lowe_ratio, max_Threshold):

        AllMatches = self.get_Allpossible_Match(featuresA,featuresB);
        valid_matches = self.All_validmatches(AllMatches,lowe_ratio)

        if len(valid_matches) > 4:
            # construct the two sets of points
            pointsA = np.float32([KeypointsA[i] for (_,i) in valid_matches])
            pointsB = np.float32([KeypointsB[i] for (i,_) in valid_matches])

            (Homograpgy, status) = self.Compute_Homography(pointsA, pointsB, max_Threshold)

            return (valid_matches, Homograpgy, status)
        else:
            return None

    def get_image_dimension(self,image):
        (h,w) = image.shape[:2]
        return (h,w)

    def get_points(self,imageA,imageB):

        (hA, wA) = self.get_image_dimension(imageA)
        (hB, wB) = self.get_image_dimension(imageB)
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        return vis


    def draw_Matches(self, imageA, imageB, KeypointsA, KeypointsB, matches, status):

        (hA,wA) = self.get_image_dimension(imageA)
        vis = self.get_points(imageA,imageB)

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(KeypointsA[queryIdx][0]), int(KeypointsA[queryIdx][1]))
                ptB = (int(KeypointsB[trainIdx][0]) + wA, int(KeypointsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis