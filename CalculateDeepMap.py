# coding: utf-8

# Author: ChanLo
# Date: 2019.4
# Reference: https://albertarmea.com/post/opencv-stereo-camera/

import cv2
import numpy as np
import sys

remap_interpolation = cv2.INTER_LINEAR
deep_visualization_scale = 2048

calibration = np.load('./calibration.npz', allow_pickle=False)
imageSize = tuple(calibration['imageSize'])
mapXL = calibration['mapXL']
print(mapXL)
mapYL = calibration['mapYL']
roiL = tuple(calibration['roiL'])
mapXR = calibration['mapXR']
mapYR = calibration['mapYR']
roiR = tuple(calibration['roiR'])

'''
videoCapture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
camera_width = 1280
camera_height = 720
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width*2)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setROI1(roiL)
stereoMatcher.setROI2(roiR)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)

while(True):
    if not videoCapture.grab():
        print("No more frames")
        break
    ret, frame = videoCapture.read()
    frameL = frame[:, camera_width:, :]
    frameR = frame[:, :camera_width, :]
    cv2.imshow('Normal', np.hstack([frameL, frameR]))

    fixedL = cv2.remap(frameL, mapXL, mapYL, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    fixedR = cv2.remap(frameR, mapXR, mapYR, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    grayFixedL = cv2.cvtColor(fixedL, cv2.COLOR_BGR2GRAY)
    grayFixedR = cv2.cvtColor(fixedR, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayFixedL, grayFixedR)
    cv2.imshow('fixedLeft', fixedL)
    cv2.imshow('fixedRight', fixedR)
    cv2.imshow('depth', depth / deep_visualization_scale)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
'''


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
'''

'''
# Filtering
kernel= np.ones((3,3),np.uint8)

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
camera_width = 1280
camera_height = 720
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width*2)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

while(videoCapture.isOpened()):
    ret, frame = videoCapture.read()
    if ret == True:
        frameL = frame[:, camera_width:, :]
        frameR = frame[:, :camera_width, :]
        # Rectify the images on rotation and alignement
        leftNice= cv2.remap(frameL, mapXL, mapYL, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
        rightNice= cv2.remap(frameR, mapXR, mapYR, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

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

'''
from matplotlib import pyplot as plt

imgL = cv2.imread('./image/L3.png',0)
imgR = cv2.imread('./image/R3.png',0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()

'''