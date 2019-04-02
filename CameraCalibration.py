# coding: utf-8

# Author: ChanLo
# Date: 2019.3
# Reference: https://github.com/LearnTechWithUs/Stereo-Vision

import cv2
import numpy as np

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteriaStereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Object points
objPoints = np.zeros((9*7, 3), np.float32)
objPoints[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)


videoCapture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
'''
print('Showing camera feed. Click window or press any key to stop.')
while(videoCapture.isOpened()):
    fps = 30 # an assumption
    # size = (1280, 480)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), \
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ret, frame = videoCapture.read()
    if ret == True:
        frameL = frame[:, 640:, :]
        frameR = frame[:, :640, :]
        cv2.imshow('frame',frameL)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    else:
        break
'''
videoCapture.release()
cv2.destroyAllWindows()  