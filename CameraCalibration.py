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

# Camera Calibration
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

# Determine the new values for different patameters
# Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objPoints, imagePointsL, ChessImageL.shape[::-1], None, None)
hL, wL = ChessImageL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

# Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objPoints, imagePointsR, ChessImageR.shape[::-1], None, None)
hR, wR = ChessImageR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

print("Camera is ready to use.")

# Calibrate the Camera for Stereo
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints,
                                                        imagePointsL,
                                                        imagePointsR,
                                                        mtxL,
                                                        distL,
                                                        mtxR,
                                                        distR,
                                                        ChessImageL.shape[::-1],
                                                        criteriaStereo,
                                                        flags)

rectifyScale = 0
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImageL.shape[::-1], R, T, rectifyScale, (0, 0))
LeftStereoMap = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImageL.shape[::-1], cv2.CV_16SC2)
RightStereoMap = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImageR.shape[::-1], cv2.CV_16SC2)

# Parameters for the StereoVision
# Create StereoSGBM and prepare all parameters
windowSize = 3
minDisp = 2
numDisp = 130 - minDisp
stereo = cv2.StereoSGBM_create(minDisparity=minDisp,
                                numDisparities=numDisp,
                                blockSize=windowSize,
                                P1=8*3*windowSize**2,
                                P2=32*3*windowSize**2,
                                disp12MaxDiff=5,
                                uniquenessRatio=10,
                                speckleWindowSize=100,
                                speckleRange=32)

# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wlsFilter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=strereo)
wlsFilter.setLambda(lmbda)
wlsFilter.setSigmaColor(sigma)

# Starting the StereoVision

# Call the camera
videoCapture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)


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

videoCapture.release()
cv2.destroyAllWindows()  
