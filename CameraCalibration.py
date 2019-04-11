# coding: utf-8

# Author: ChanLo
# Date: 2019.3
# Reference: https://github.com/LearnTechWithUs/Stereo-Vision

import cv2
import numpy as np

# Filtering
kernel= np.ones((3,3),np.uint8)

def coordsMouseDisp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print(x, y, disp[y, x], filteredImg[y, x])
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y + u, x + v]
        average = average / 9
        distance = -593.97 * average**(3) + 1506.8 * average**(2) - 1373.1 * average + 522.06
        distance = np.around(distance * 0.01, decimals=2)
        print('Distance: ' + str(distance) + ' m')


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
print("Starting calibration for the camera...")
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
'''
print("retL: ", retL)
print("mtxL: ", mtxL) # 内参数矩阵
print("distL: ", distL) # 畸变系数
print("rvecsL: ", rvecsL) # 旋转向量
print("tvecsL: ", tvecsL) # 平移向量
'''
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
leftStereoMap = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImageL.shape[::-1], cv2.CV_16SC2)
rightStereoMap = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImageR.shape[::-1], cv2.CV_16SC2)

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

wlsFilter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wlsFilter.setLambda(lmbda)
wlsFilter.setSigmaColor(sigma)

# Starting the StereoVision

# Call the camera
videoCapture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)

while(videoCapture.isOpened()):
    ret, frame = videoCapture.read()
    if ret == True:
        frameL = frame[:, 640:, :]
        frameR = frame[:, :640, :]
        # Rectify the images on rotation and alignement
        leftNice= cv2.remap(frameL, leftStereoMap[0], leftStereoMap[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
        rightNice= cv2.remap(frameR, rightStereoMap[0], rightStereoMap[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        '''
        # Draw red lines
        for line in range(0, int(leftNice.shape[0] / 20)):
            leftNice[line * 20, :] = (0, 0, 255)
            rightNice[line * 20, :] = (0, 0, 255)
        
        for line in range(0, int(frameL.shape[0] / 20)):
            frameL[line * 20, :] = (0, 255, 0)
            frameR[line * 20, :] = (0, 255, 0)
        '''
        # Show the undistorted images
        cv2.imshow('Both Images', np.hstack([leftNice, rightNice]))
        cv2.imshow('Normal', np.hstack([frameL, frameR]))

        # Convert from color(BGR) to gray
        grayR = cv2.cvtColor(rightNice, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(leftNice, cv2.COLOR_BGR2GRAY)

        # Compute the 2 images for the DepthImage
        disp = stereo.compute(grayL, grayR)#.astype(np.float32)/ 16
        dispL = disp
        dispR = stereoR.compute(grayR,grayL)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)

        # Using the WLS filter
        filteredImg = wlsFilter.filter(dispL, grayL, None, dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)
        #cv2.imshow('Disparity Map', filteredImg)
        disp = ((disp.astype(np.float32) / 16) - minDisp) / numDisp # Calculation allowing us to have 0 for the most distant object able to detect

        '''
        # Resize the image for faster executions
        dispR= cv2.resize(disp, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        '''

        # Filtering the Results with a closing filter
        closing= cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

        # Colors map
        dispc = (closing - closing.min()) * 255
        dispC = dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
        dispColor= cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
        filtColor= cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN) 

        # Show the result for the Depth_image
        #cv2.imshow('Disparity', disp)
        #cv2.imshow('Closing', closing)
        #cv2.imshow('Color Depth', dispColor)
        cv2.imshow('Filtered Color Depth', filtColor)

        # Mouse click
        cv2.setMouseCallback("Filtered Color Depth", coordsMouseDisp, filtColor)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    else:
        break

videoCapture.release()
cv2.destroyAllWindows()  
