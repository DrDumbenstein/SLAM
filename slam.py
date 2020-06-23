import cv2
import numpy as np
from display import Display
from frame import Frame, denormalize, match
import g2o


# Intrinsic Parameters Of The Camera
W = 1920//2
H = 1080//2
F = 200
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))

# Main classes
display = Display(W, H)

frames = []


def process_frame(img):
    img = cv2.resize(img, (H, W))
    frame = Frame(img, K)
    frames.append(frame)

    if len(frames) <=1:
        return

    ret, Rt = match(frames[-2], frames[-1])

    for pt1, pt2 in ret:
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
