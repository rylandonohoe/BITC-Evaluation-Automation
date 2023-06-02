import cv2 as cv
import numpy as np
import sys

def image_acquisition(file_path):
    img = cv.imread(cv.samples.findFile(file_path))

    if img is None:
        sys.exit("Could not read the image.") # make sure image is png, jpg, or jpeg (some other file types could work as well)

    cv.imshow("Display window", img)
    k = cv.waitKey(0)

    return img

def image_pre_processing(img):
    # grayscale conversion
    gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # image is now 1-channel

    cv.imshow("Display window", gray1)
    k = cv.waitKey(0)

    # noise reduction (part 1)
    blur1 = cv.medianBlur(gray1, 3)

    cv.imshow("Display window", blur1)
    k = cv.waitKey(0)

    # thresholding (part 1)
    ret, thresh1 = cv.threshold(blur1, 200, 255, cv.THRESH_BINARY)

    cv.imshow("Display window", thresh1)
    k = cv.waitKey(0)

    # edge detection (part 1)
    lower_threshold = 100 # lower threshold value in Hysteresis Thresholding
    upper_threshold = 200 # upper threshold value in Hysteresis Thresholding
    aperture_size = 3 # aperture size of the Sobel filter
    edges1 = cv.Canny(thresh1, lower_threshold, upper_threshold, aperture_size)

    cv.imshow("Display window", edges1)
    k = cv.waitKey(0)

    # dilation (part 1)
    element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilated1 = cv.dilate(edges1, element, iterations=1)

    cv.imshow("Display window", dilated1)
    k = cv.waitKey(0)

    # erosion
    element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    eroded = cv.erode(dilated1, element, iterations=1)

    cv.imshow("Display window", eroded)
    k = cv.waitKey(0)

    # noise reduction (part 2)
    blur2 = cv.fastNlMeansDenoising(eroded, None, h=30, templateWindowSize=30, searchWindowSize=30)
    
    cv.imshow("Display window", blur2)
    k = cv.waitKey(0)

    # dilation (part 2)
    element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilated2 = cv.dilate(blur2, element, iterations=1)

    cv.imshow("Display window", dilated2)
    k = cv.waitKey(0)

    # thresholding (part 2)
    ret, thresh2 = cv.threshold(dilated2, 25, 255, cv.THRESH_BINARY)

    cv.imshow("Display window", thresh2)
    k = cv.waitKey(0)

    # erosion (part 2)
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    eroded2 = cv.erode(thresh2, element, iterations=1)

    cv.imshow("Display window", eroded2)
    k = cv.waitKey(0)

    pre_processed_img = eroded2

    return pre_processed_img





def process_image(file_path):
    img = image_acquisition(file_path)
    pre_processed_img = image_pre_processing(img)



    return StarC_C1, StarC_C2, StarC_C3, StarC_C4, StarC_C5, StarC_C6, StarC, StarC_SV, StarC_HCoC, StarC_VCoC

file_path = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/patients/Braun/StarC.png"
print(process_image(file_path))