# coding: utf-8

# Author: ChanLo
# Date: 2019.3
# Reference: https://github.com/LearnTechWithUs/Stereo-Vision

import cv2
import numpy as np

id_image=0

# termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

videoCapture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)

while(videoCapture.isOpened()):
    ret, frame = videoCapture.read()
    frameL = frame[:, 640:, :]
    frameR = frame[:, :640, :]
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    cv2.imshow('imgL', frameL)
    cv2.imshow('imgR', frameR)

    # Find the chess board corners
    retR, cornersR = cv2.findChessboardCorners(grayR,(9,7),None)  # Define the number of chess corners (here 9 by 6) we are looking for with the right Camera
    retL, cornersL = cv2.findChessboardCorners(grayL,(9,7),None)

    # If found, add object points, image points (after refining them)
    if (retR == True) & (retL == True):
        corners2R= cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)    # Refining the Position
        corners2L= cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(grayR,(9,7),corners2R,retR)
        cv2.drawChessboardCorners(grayL,(9,7),corners2L,retL)
        cv2.imshow('VideoR',grayR)
        cv2.imshow('VideoL',grayL)
        if cv2.waitKey(0) & 0xFF == ord('s'):   # Push "s" to save the images and "c" if you don't want to
            str_id_image= str(id_image)
            print('Images ' + str_id_image + ' saved for right and left cameras')
            cv2.imwrite('./image/chessboard-R'+str_id_image+'.png',frameR) # Save the image in the file where this Programm is located
            cv2.imwrite('./image/chessboard-L'+str_id_image+'.png',frameL)
            id_image=id_image+1
        else:
            print('Images not saved')
        
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Quit.')
        break

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