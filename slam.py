import cv2
import numpy as np
from display import Display

W = 1920//2
H = 1080//2

display = Display(W, H)


def process_frame(img):
    # resize
    img = cv2.resize(img, (H, W))
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
