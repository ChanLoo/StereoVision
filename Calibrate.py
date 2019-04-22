# coding: utf-8

# Author: ChanLo
# Date: 2019.4
# Reference: https://albertarmea.com/post/opencv-stereo-camera/

import cv2
import numpy as np

id_image = 0
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
max_images = 64
