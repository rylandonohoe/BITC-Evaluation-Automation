import cv2 as cv
import numpy as np
import sys
from sklearn.cluster import DBSCAN

def image_acquisition(file_path):
    img = cv.imread(cv.samples.findFile(file_path))

    if img is None:
        sys.exit("Could not read the image.") # make sure image is png, jpg, or jpeg (some other file types could work as well)

    #cv.imshow("Display window", img)
    #k = cv.waitKey(0)

    return img

def image_pre_processing(img):
    # grayscale conversion
    gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # image is now 1-channel

    #cv.imshow("Display window", gray1)
    #k = cv.waitKey(0)

    # thresholding (part 1)
    ret, thresh1 = cv.threshold(gray1, 127, 255, cv.THRESH_BINARY)

    #cv.imshow("Display window", thresh1)
    #k = cv.waitKey(0)

    # edge detection (part 1)
    lower_threshold = 25 # lower threshold value in Hysteresis Thresholding
    upper_threshold = 50 # upper threshold value in Hysteresis Thresholding
    aperture_size = 3 # aperture size of the Sobel filter
    edges1 = cv.Canny(thresh1, lower_threshold, upper_threshold, aperture_size)

    #cv.imshow("Display window", edges1)
    #k = cv.waitKey(0)

    # noise reduction (part 1)
    blur1 = cv.fastNlMeansDenoising(edges1, None, h=20, templateWindowSize=25, searchWindowSize=25)
    
    #cv.imshow("Display window", blur1)
    #k = cv.waitKey(0)

    # dilation (part 1)
    element = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    dilated1 = cv.dilate(blur1, element, iterations=1)

    #cv.imshow("Display window", dilated1)
    #k = cv.waitKey(0)

    # thresholding (part 2)
    ret, thresh2 = cv.threshold(dilated1, 127, 255, cv.THRESH_BINARY)

    #cv.imshow("Display window", thresh2)
    #k = cv.waitKey(0)

    # edge detection (part 2)
    lower_threshold = 300 # lower threshold value in Hysteresis Thresholding
    upper_threshold = 400 # upper threshold value in Hysteresis Thresholding
    aperture_size = 3 # aperture size of the Sobel filter
    edges2 = cv.Canny(thresh2, lower_threshold, upper_threshold, aperture_size)

    #cv.imshow("Display window", edges2)
    #k = cv.waitKey(0)

    # dilation (part 2)
    element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilated2 = cv.dilate(edges2, element, iterations=1)

    #cv.imshow("Display window", dilated2)
    #k = cv.waitKey(0)

    # thresholding (part 3)
    ret, thresh3 = cv.threshold(dilated2, 10, 255, cv.THRESH_BINARY)

    #cv.imshow("Display window", thresh3)
    #k = cv.waitKey(0)

    pre_processed_img = thresh3

    return pre_processed_img

def contour_detection(img, pre_processed_img):
    contours, hierarchy = cv.findContours(pre_processed_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    contour_img = img.copy()
    cv.drawContours(contour_img, contours, -1, (0, 255, 0), 5)

    #cv.imshow("Display window", contour_img)
    #k = cv.waitKey(0)

    return contour_img, contours

def centroid_detection(img, contours):
    centroids = []
    arrow_centroid = None

    height, width = img.shape[:2]
    border_buffer = 50
    min_distance_to_border = float('inf')
    for contour in contours:
        M = cv.moments(contour)
        area = cv.contourArea(contour)
        if M['m00'] != 0 and 1000 < area < 5000:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            distance_to_border = min(cx, cy, width - cx, height - cy)
            if distance_to_border > border_buffer: # avoid border contours caused by scanning
                if distance_to_border < min_distance_to_border: # isolate arrow centroid
                    min_distance_to_border = distance_to_border
                    if arrow_centroid is not None:
                        centroids.append([arrow_centroid[0], arrow_centroid[1]])
                    arrow_centroid = cx, cy
                else:
                    centroids.append([cx, cy])

    merged_centroids = []
    centroids_array = np.array(centroids)
    clustering = DBSCAN(eps=100, min_samples=1).fit(centroids_array) 
    labels = clustering.labels_
    
    for label in set(labels):
        if label != -1: # ignore noise points
            cluster_points = centroids_array[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            merged_centroids.append(tuple(centroid))

    centroid_img = img.copy()
    centroid_thickness = 8
    for centroid in merged_centroids:
        cv.circle(centroid_img, (int(centroid[0]), int(centroid[1])), centroid_thickness, (0, 0, 255), -1)
    cv.circle(centroid_img, arrow_centroid, centroid_thickness, (0, 0, 0), -1)

    #cv.imshow("Display window", centroid_img)
    #k = cv.waitKey(0)

    return centroid_img, merged_centroids, arrow_centroid

def orient_image(centroid_img, merged_centroids, arrow_centroid):
    # determine side arrow is on assuming arrow is centred
    def get_closest_side(img, arrow_centroid):
        height, width = img.shape[:2]
        x, y = arrow_centroid

        if x <= width/4:
            return "left"
        elif x >= 3*width/4:
            return "right"
        elif y <= height/4:
            return "top"
        elif y >= 3*height/4:
            return "bottom"
        else:
            return None
    
    def rotation_based_on_side(side):
        return {"top": 180, "right": 270, "bottom": 0, "left": 90}.get(side, 0)

    def rotate_image_and_centroids(img, centroids, angle):
        height, width = img.shape[:2]
        centre = (width / 2, height / 2)

        M = cv.getRotationMatrix2D(centre, angle, 1.0)

        # calculate new image size after rotation
        radians = np.deg2rad(angle)
        new_width = int(abs(height * np.sin(radians)) + abs(width * np.cos(radians)))
        new_height = int(abs(height * np.cos(radians)) + abs(width * np.sin(radians)))

        M[0, 2] += new_width / 2 - centre[0]
        M[1, 2] += new_height / 2 - centre[1]

        rotated_img = cv.warpAffine(img, M, (new_width, new_height))

        # rotate centroids
        centroids = np.array(centroids)
        centroids = np.hstack((centroids, np.ones((centroids.shape[0], 1))))
        rotated_centroids = np.matmul(M, centroids.T).T

        return rotated_img, rotated_centroids

    closest_side = get_closest_side(centroid_img, arrow_centroid)
    angle = rotation_based_on_side(closest_side)
    rotated_img, rotated_centroids = rotate_image_and_centroids(centroid_img, merged_centroids, angle)
    
    #cv.imshow("Display window", rotated_img)
    #k = cv.waitKey(0)

    # remove the centre four centroids from the final list (keeping them in simply makes the values less sensitive relative to one another)
    LineC_T_C1 = []
    for centroid in rotated_centroids:
        height, width = rotated_img.shape[:2]
        x, y = centroid
        if not ((width/2 - 200) <= x <= (width/2 + 200)):
            LineC_T_C1.append(centroid)

    CoC_target_img = rotated_img.copy()
    centroid_thickness = 8
    for centroid in LineC_T_C1:
            cv.circle(CoC_target_img, (int(centroid[0]), int(centroid[1])), centroid_thickness, (0, 0, 0), -1)

    #cv.imshow("Display window", CoC_target_img)
    #k = cv.waitKey(0)

    return CoC_target_img, LineC_T_C1

def process_image(file_path):
    img = image_acquisition(file_path)
    pre_processed_img = image_pre_processing(img)
    contour_img, contours = contour_detection(img, pre_processed_img)
    centroid_img, merged_centroids, arrow_centroid = centroid_detection(img, contours)
    CoC_target_img, LineC_T_C1 = orient_image(centroid_img, merged_centroids, arrow_centroid)
    return LineC_T_C1

#file_path = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/templates/LineC_T.png"
#print(process_image(file_path))