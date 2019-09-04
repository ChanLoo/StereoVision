# coding: utf-8

# Author: ChanLo
# Date: 2019.9
# Reference: https://blog.csdn.net/mangobar/article/details/88556242

import cv2
import numpy as np


def callbackFunc(event, x, y, flags, param):
    """Add click event

    Add click event to print distance of current point.

    """
    if event == cv2.EVENT_LBUTTONDOWN:        
        print(threeD[y][x])

cv2.setMouseCallback("depth", callbackFunc, None)

calibration = np.load('./calibration.npz', allow_pickle=False)
imageSize = tuple(calibration['imageSize'])
mapXL = calibration['mapXL']
print(mapXL)
mapYL = calibration['mapYL']
roiL = tuple(calibration['roiL'])
mapXR = calibration['mapXR']
mapYR = calibration['mapYR']
roiR = tuple(calibration['roiR'])
Q = calibration['Q']

# Parameters for the StereoVision
# Create StereoSGBM and prepare all parameters
windowSize = 3
minDisp = 2
numDisp = 130 - minDisp

# Call the camera.
videoCapture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
camera_width = 1280
camera_height = 720
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width*2)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

while(videoCapture.isOpened()):
    ret, frame = videoCapture.read()
    if not ret:
        break
    frameL = frame[:, camera_width:, :]
    frameR = frame[:, :camera_width, :]
    # Rectify the images on rotation and alignement.
    leftNice= cv2.remap(frameL, mapXL, mapYL, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    rightNice= cv2.remap(frameR, mapXR, mapYR, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    # Convert the images from color(BGR) to gray.
    grayL = cv2.cvtColor(leftNice, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rightNice, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoSGBM_create(minDisparity=minDisp,
                                numDisparities=numDisp,
                                blockSize=windowSize,
                                P1=8*3*windowSize**2,
                                P2=32*3*windowSize**2,
                                disp12MaxDiff=5,
                                uniquenessRatio=10,
                                speckleWindowSize=100,
                                speckleRange=32)
    # Compute the 2 images for the DepthImage
    disparity = stereo.compute(grayL, grayR)#.astype(np.float32)/ 16
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Extend the image into 3d space with the value of the current distance in the z direction.
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)
    print(type(threeD))
    print(threeD[360][320])

    cv2.imshow("Left", leftNice)
    cv2.imshow("Right", rightNice)
    cv2.imshow("Depth", disp)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows() 
