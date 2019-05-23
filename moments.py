from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
import random as rng
from pprint import pprint
rng.seed(12345)

video_capture = cv.VideoCapture(0)
def thresh_callback(val, src_gray):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)


    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    # Get the moments
    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv.moments(contours[i])

    # Get the mass centers
    mc = [None]*len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    # Draw contours

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2)
        cv.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)


    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for i in range(len(contours)):
        if(len(contours[i]) < 50):
            continue
        else:
            # cv.imshow('Contours'+str(i), )
            # minx = max(contours[i])
            # pprint(max(contours[i][0]))
            c = contours[i]
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            topLeft = (extLeft[0], extTop[1])
            botRight = (extRight[0], extBot[1])
            cv.circle(drawing, topLeft, 8, (255, 255, 255), -1)
            cv.circle(drawing, botRight, 8, (255, 255, 255), -1)

            # pprint(src_gray[topLeft[0]:botRight[0], topLeft[1]:botRight[1]])
            if (len(src_gray[topLeft[0]:botRight[0], topLeft[1]:botRight[1]]) > 0):
                cv.imshow('Contours2', src_gray[topLeft[1]:botRight[1] , topLeft[0]:botRight[0]])
                cv.waitKey(100)



            print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))


    cv.imshow('Contours', drawing)
    cv.waitKey(200)

def readVideo():

    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, src = video_capture.read()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3,3))
        source_window = 'Source'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        max_thresh = 255
        thresh = 100 # initial threshold
        cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
        thresh_callback(thresh, src_gray)




# src = cv.imread("coins.jpg", cv.IMREAD_COLOR)
#
# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# src_gray = cv.blur(src_gray, (3,3))
# source_window = 'Source'
# cv.namedWindow(source_window)
# cv.imshow(source_window, src)
# max_thresh = 255
# thresh = 100 # initial threshold
# cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
# thresh_callback(thresh)

readVideo()

while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord("c"):
        break

cv.destroyAllWindows()