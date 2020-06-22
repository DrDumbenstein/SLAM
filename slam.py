import cv2
import numpy as np
from display import Display
from extractor import Extractor

W = 1920//2
H = 1080//2
F = 1

# Intrinsic Parameters Of The Camera
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))

display = Display(W, H)

fe = Extractor(K)


def process_frame(img):
    img = cv2.resize(img, (H, W))
    ret = fe.extract(img)

    for pt1, pt2 in ret:
        u1, v1 = fe.denormalize(pt1)
        u2, v2 = fe.denormalize(pt2)

        cv2.circle(img, (u1, v1), color=(0, 0, 255), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(0, 255, 0))

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
