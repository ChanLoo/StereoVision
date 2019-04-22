import cv2

videoCapture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
camera_width = 1280
camera_height = 720
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width*2)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
id_image = 0

while(True):
    if not videoCapture.grab():
        print("No more frames")
        break
    ret, frame = videoCapture.read()
    frameL = frame[:, camera_width:, :]
    frameR = frame[:, :camera_width, :]
    cv2.imshow('imgL', frameL)
    cv2.imshow('imgR', frameR)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        str_id_image = str(id_image)
        print('Images ' + str_id_image + ' saved for left and right cameras.')
        cv2.imwrite('./image/L' + str_id_image + '.png', frameL)
        cv2.imwrite('./image/R' + str_id_image + '.png', frameR)
        id_image = id_image + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Quit.')
        break

videoCapture.release()
cv2.destroyAllWindows()