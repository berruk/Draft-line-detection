import os
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt


img = "example.jpg"
img = cv.imread(img)

#Extracting Red Channel
red = img.copy()
red[:, :, 0] = 0
red[:, :, 1] = 0

ret,img = cv.threshold(red,127,255,cv.THRESH_BINARY)

#Convert to Gray and threshold
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img,0,200,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

#Extract lines
thresh = cv.Canny(thresh,100,200)
cdst = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)


#Draw lines with Hough

lines = cv.HoughLines(thresh, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)


#Draw lines with probabilistic Hough
linesP = cv.HoughLinesP(thresh, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)



cv.namedWindow("output", cv.WINDOW_NORMAL)
cv.resizeWindow("output", 800, 600)
cv.imshow("output", cdst)

cv.waitKey()
