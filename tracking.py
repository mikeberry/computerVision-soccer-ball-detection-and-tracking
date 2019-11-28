import cv2
import numpy as np


def render_with_tracking_info(img):
    return img



if __name__ == "__main__":

    cap = cv2.VideoCapture('soccer-ball.mp4')
    while True:
        ret, frame = cap.read()
        if frame is not None:
            result = render_with_tracking_info(frame)
            cv2.imshow('frame', result)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
