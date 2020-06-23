import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def extractRt(E):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal() < 0):
        R = np.dot(np.dot(U, W.T), Vt)

    t = U[:, 2]
    pose = np.concatenate((R, t.reshape(3, 1)), axis=1)
    return pose


def match(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # Matching
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test
    ret = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            ret.append((p1, p2))

    assert len(ret) !=0
    ret = np.array(ret)

    # fit matrix
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            EssentialMatrixTransform,
                            # FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.005, max_trials=100)

    # Ignore outliers
    ret = ret[inliers]
    Rt = extractRt(model.params)

    return ret, Rt


def extract(img):
    orb = cv2.ORB_create(5)
    # Detection
    features = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), maxCorners=3000, qualityLevel=0.01,
                                       minDistance=3)
    # Extraction
    kps = [cv2.KeyPoint(crd[0][0], crd[0][1], 20) for crd in features]
    kps, des = orb.compute(img, kps)

    # Return points and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def normalize(Kinv, pt):
    return np.dot(Kinv, add_ones(pt).T).T[:, 0:2]


def denormalize(K, pt):
    ret = np.array(np.dot(K, [pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


class Frame(object):
    def __init__(self, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        features, self.des = extract(img)
        self.pts = normalize(self.Kinv, features)







