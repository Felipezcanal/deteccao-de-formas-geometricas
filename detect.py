import cv2
import sys
import numpy as np
import logging as log
import datetime as dt
from pprint import pprint
from time import sleep


video_capture = cv2.VideoCapture(0)


cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('gray', cv2.WINDOW_KEEPRATIO)

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Display the resulting frame
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    for c in contours:
        pprint(len(c))
        if(len(c) > 50):
            cv2.drawContours(frame, c, -1,(0, 255, 0), 5)
    cv2.imshow('gray', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
