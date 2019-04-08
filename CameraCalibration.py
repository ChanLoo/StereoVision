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
preObjPoints = np.zeros((9*7, 3), np.float32)
preObjPoints[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

objPoints = []
imagePointsL = []
imagePointsR = []

# Calibration
print("Starting calibration for the cameras...")
# call all saved images
for i in range(0, 50):
    t = str(i)
    ChessImageL = cv2.imread('./image/chessboard-L' + t + '.png', 0)
    ChessImageR = cv2.imread('./image/chessboard-R' + t + '.png', 0)
    retL, cornersL = cv2.findChessboardCorners(ChessImageL, (9, 7), None)
    retR, cornersR = cv2.findChessboardCorners(ChessImageR, (9, 7), None)
    if retL == True & retR == True:
        objPoints.append(preObjPoints)
        cv2.cornerSubPix(ChessImageL, cornersL, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImageR, cornersR, (11, 11), (-1, -1), criteria)
        imagePointsL.append(cornersL)
        imagePointsR.append(cornersR)

