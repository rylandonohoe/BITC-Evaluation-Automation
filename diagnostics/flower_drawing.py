import cv2 as cv
import numpy as np
import sys

def image_acquisition(file_path):
    img = cv.imread(cv.samples.findFile(file_path))

    if img is None:
        sys.exit("Could not read the image.") # make sure image is png, jpg, or jpeg (some other file types could work as well)

    cv.imshow("img", img)
    k = cv.waitKey(0)
    cv.destroyWindow("img")

    # isolate flower
    height, width = img.shape[:2]
    resized = img[int(height/3*2):int(height - 5), int(width/2 + 15):int(width - 45)]

    cv.imshow("resized", resized)
    k = cv.waitKey(0)
    cv.destroyWindow("resized")

    return resized

def image_pre_processing(resized):
    # grayscale conversion
    gray1 = cv.cvtColor(resized, cv.COLOR_BGR2GRAY) # image is now 1-channel

    cv.imshow("gray1", gray1)
    k = cv.waitKey(0)
    cv.destroyWindow("gray1")

    # thresholding
    ret, thresh = cv.threshold(gray1, 225, 255, cv.THRESH_BINARY)

    cv.imshow("thresh", thresh)
    k = cv.waitKey(0)
    cv.destroyWindow("thresh")

    pre_processed_img = thresh
    
    return pre_processed_img

def corner_detection(resized, pre_processed_img):
    # corner detection
    blockSize = 50 # size of neighbourhood considered for corner detection
    kSize = 15 # aperture parameter of the Sobel derivative used
    k = 0.2 # Harris detector free parameter in the equation
    dst = cv.cornerHarris(pre_processed_img, blockSize, kSize, k)
    corners = np.where(dst > 0.02 * dst.max())
    
    corner_img = resized.copy()
    corner_img[corners] = [0, 255, 0]

    cv.imshow("corner_img", corner_img)
    k = cv.waitKey(0)
    cv.destroyWindow("corner_img")
    
    return corner_img, corners



def process_image(file_path):
    resized = image_acquisition(file_path)
    pre_processed_img = image_pre_processing(resized)
    corner_img, corners = corner_detection(resized, pre_processed_img)



    #return DrawFlower_F, DrawFlower_D, DrawFlower_A, DrawFlower, DrawFlower_SV



for name in ["Braun", "BW", "Daskalon", "Franz", "Gerke", "Kuhn", "Loffelad", "Sigruner"]:
    file_path = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/patients/" + name + "/Draw.png"
    print(process_image(file_path))