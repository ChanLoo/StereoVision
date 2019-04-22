import cv2
import numpy as np
from matplotlib import pyplot as plt

imgL = cv2.imread('./image/chessboard-L1.png',0)
imgR = cv2.imread('./image/chessboard-R1.png',0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()


'''
videoCapture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
camera_width = 1280
camera_height = 720
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width*2)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

while(True):
    if not videoCapture.grab():
        print("No more frames")
        break
    ret, frame = videoCapture.read()
    frameL = frame[:, camera_width:, :]
    frameR = frame[:, :camera_width, :]
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    cv2.imshow('imgL', frameL)
    cv2.imshow('imgR', frameR)
    
    #_, leftFrame = left.retrieve()
    #_, rightFrame = right.retrieve()

    #cv2.imshow('left', leftFrame)
    #cv2.imshow('right', rightFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
'''