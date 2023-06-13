import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
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
    resized = img[int(height/3*2 + 25):int(height - 5), int(width/2 + 15):int(width - 45)]

    cv.imshow("resized", resized)
    k = cv.waitKey(0)
    cv.destroyWindow("resized")

    return resized

def process_image(file_path):
    resized = image_acquisition(file_path)



    #return DrawFlower_F, DrawFlower_D, DrawFlower_A, DrawFlower, DrawFlower_SV



#for name in ["Braun", "BW", "Daskalon", "Dotzamer", "Franz", "Gerke", "Kuhn", "Loffelad", "Sigruner"]:
    #file_path = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/patients/" + name + "/Draw.png"
    #print(process_image(file_path))