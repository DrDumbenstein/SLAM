import cv2
import numpy as np
from display import Display
from frame import Frame, denormalize, match_frames, IRt
import g2o


# Intrinsic Parameters Of The Camera
W = 1920//2
H = 1080//2
F = 200
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))

# Main classes
display = Display(W, H)
frames = []


class Point(object):
    # Point in the world
    # Each point is observed in multiple frames

    def __init__(self, loc):
        self.location = loc
        self.frames = []
        self.idxs = []

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def triangulate(pose1, pose2, pt1, pt2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pt1.T, pt2.T).T


def process_frame(img):
    img = cv2.resize(img, (H, W))
    frame = Frame(img, K)
    frames.append(frame)
    if len(frames) <=1:
        return

    idx1, idx2, Rt = match_frames(frames[-1], frames[-2])
    frames[-1].pose = np.dot(Rt, frames[-2].pose)
    pts4d = triangulate(frames[-1].pose, frames[-2].pose, frames[-1].pts[idx1], frames[-2].pts[idx2])
    # Homogenous 3D points
    pts4d /= pts4d[:, 3:]

    # Reject points without enough parallox
    # Reject the points behind the camera
    good_4dpts = np.abs(pts4d[:, 3] > 0.005) & (pts4d[:, 2] > 0)
    pts4d = pts4d[good_4dpts]
    print(sum(good_4dpts), len(good_4dpts))

    for i,p in enumerate(pts4d):
        if not good_4dpts[i]:
            continue
        pt = Point(p)
        pt.add_observation(frames[-1], idx1[i])
        pt.add_observation(frames[-2], idx2[i])

    for pt1, pt2 in zip(frames[-1].pts[idx1], frames[-2].pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    display.draw(img)


if __name__ == "__main__":
    capture = cv2.VideoCapture('carvid.mp4')
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(fps)
    while capture.isOpened():
        # Capture frame by frame
        ret, frame = capture.read()
        if ret:
            # Rotating and Flipping
            frame = np.fliplr(frame)
            frame = np.rot90(frame)
            # From BRG to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            process_frame(frame)

        else:
            break
