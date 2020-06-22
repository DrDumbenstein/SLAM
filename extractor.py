import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


class Extractor(object):

    def __init__(self, K):
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def normalize(self, pt):
        return np.dot(self.Kinv, add_ones(pt).T).T[:, 0:2]

    def denormalize(self, pt):
        ret = np.array(np.dot(self.K, [pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))

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
            # Normalize coors: substract to move to 0
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            print(ret.shape)
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    #EssentialMatrixTransform,
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=0.01, max_trials=100)

            ret = ret[inliers]
            s, v, d = np.linalg.svd(model.params)
            print(v)

        self.last = {'kps': kps, 'des': des}

        return ret
