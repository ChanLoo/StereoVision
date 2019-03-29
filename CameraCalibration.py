# coding: utf-8
# author: ChanLo
# date: 2019.3.29

import cv2

videoCapture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)

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

videoCapture.release()