import cv2 as cv
import math
import numpy as np
import sys

# 1. IMAGE ACQUISITION

filename = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/diamond_drawing/test1.png"
img = cv.imread(cv.samples.findFile(filename))

if img is None:
    sys.exit("Could not read the image.") # make sure image is png, jpg, or jpeg (some other file types could work as well)

#cv.imshow("Display window", img)
#k = cv.waitKey(0)

# 2. IMAGE PRE-PROCESSING

# grayscale conversion
gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(gray1, cv.COLOR_GRAY2RGB) # convert gray1 to three channel for addWeighted function

#cv.imshow("Display window", gray3)
#k = cv.waitKey(0)

# noise reduction (Gaussian blur)
kernel_size = (3, 3) # larger = blurrier
blur_gray3 = cv.GaussianBlur(gray3, kernel_size, 0)

#cv.imshow("Display window", blur_gray3)
#k = cv.waitKey(0)

# 3. EDGE AND LINE DETECTION

