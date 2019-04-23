# coding: utf-8

# Author: ChanLo
# Date: 2019.4
# Reference: https://albertarmea.com/post/opencv-stereo-camera/


import cv2
import numpy as np
import sys
import os
import glob

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteriaStereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chessboard_size = (9, 7)

# Object points
preObjPoints = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
preObjPoints[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

imagesDir = './image'

def FindChessboards(imagesDir):
    cacheFile = imagesDir + '/chessboards.npz'
    try:
        cache = np.load(cacheFile)
        print("Loading data from cache file at " + cacheFile)
        return (list(cache['filenames']),
                list(cache['objectPoints']),
                list(cache['imagePoints']),
                tuple(cache['imageSize']))
    except IOError:
        print("Cache file not found.")
    
    filenames = []
    objPoints = []
    imagePoints = []
    imageSize = None

    print("Reading images at " + imagesDir)
    imagePaths = glob.glob(imagesDir + "/*.png")

    # Call all saved images
    for imagePath in sorted(imagePaths):
        image = cv2.imread(imagePath)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageSize = grayImage.shape[::-1]

        ret, corners = cv2.findChessboardCorners(grayImage, chessboard_size, None)
        if ret:
            filenames.append(os.path.basename(imagePath))
            objPoints.append(preObjPoints)
            cv2.cornerSubPix(grayImage, corners, (11, 11), (-1, -1), criteria)
            imagePoints.append(corners)
        
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
        cv2.imshow(imagesDir, image)

        cv2.waitKey(1)

    cv2.destroyWindow(imagesDir)

    np.savez_compressed(cacheFile,
                        filenames=filenames,
                        objectPoints=objPoints,
                        imagePoints=imagePoints,
                        imageSize=imageSize)
    return filenames, objPoints, imagePoints, imageSize

# Camera Calibration
print("Starting calibration for the camera...")

(filenamesL, objPointsL, imagePointsL, imageSizeL) = FindChessboards('./image/chessboard-L')
(filenamesR, objPointsR, imagePointsR, imageSizeR) = FindChessboards('./image/chessboard-R')

sys.exit()

# Determine the new values for different patameters
# Left Side
print("Calibrating left camera...")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objPoints, imagePointsL, ChessImageL.shape[::-1], None, None)
'''
print("retL: ", retL)
print("mtxL: ", mtxL) # 内参数矩阵
print("distL: ", distL) # 畸变系数
print("rvecsL: ", rvecsL) # 旋转向量
print("tvecsL: ", tvecsL) # 平移向量
'''

hL, wL = ChessImageL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))
print(OmtxL)

#sys.exit()

# Right Side
print("Calibrating right camera...")
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objPoints, imagePointsR, ChessImageR.shape[::-1], None, None)
hR, wR = ChessImageR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

print("Camera is ready to use.")

# Calibrate the Camera for Stereo
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

print("Calibrating cameras together...")
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints,
                                                        imagePointsL,
                                                        imagePointsR,
                                                        mtxL,
                                                        distL,
                                                        mtxR,
                                                        distR,
                                                        ChessImageL.shape[::-1],
                                                        None,
                                                        None,
                                                        None,
                                                        None,
                                                        flags,
                                                        criteriaStereo)

print("Rectifying cameras...")
rectifyScale = 0.25
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImageL.shape[::-1], R, T, None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, rectifyScale)
leftStereoMap = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImageL.shape[::-1], cv2.CV_16SC2)
rightStereoMap = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImageR.shape[::-1], cv2.CV_16SC2)
