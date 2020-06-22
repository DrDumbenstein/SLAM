import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


class Extractor(object):

    def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, img):
        # Detection
        features = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), maxCorners=3000, qualityLevel=0.01,
                                           minDistance=3)
        # Extraction
        kps = [cv2.KeyPoint(crd[0][0], crd[0][1], 20) for crd in features]
        kps, des = self.orb.compute(img, kps)

        # Matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # Filter
        matches = []
        if len(ret) > 0:
            ret = np.array(ret)
            print(ret.shape)
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    FundamentalMatrixTransform, min_samples=8,
                                    residual_threshold=0.01, max_trials=100)

            ret = ret[inliers]

        self.last = {'kps': kps, 'des': des}

        return ret
