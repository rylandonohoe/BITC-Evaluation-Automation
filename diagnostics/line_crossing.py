import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import sys

def image_acquisition(file_path):
    img = cv.imread(cv.samples.findFile(file_path))

    if img is None:
        sys.exit("Could not read the image.") # make sure image is png, jpg, or jpeg (some other file types could work as well)

    #cv.imshow("img", img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("img")

    return img

def image_pre_processing(img):
    # grayscale conversion
    gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # image is now 1-channel

    #cv.imshow("gray1", gray1)
    #k = cv.waitKey(0)
    #cv.destroyWindow("gray1")

    # thresholding (part 1)
    ret, thresh1 = cv.threshold(gray1, 225, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh1", thresh1)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh1")

    # edge detection
    lower_threshold = 10 # lower threshold value in Hysteresis Thresholding
    upper_threshold = 20 # upper threshold value in Hysteresis Thresholding
    aperture_size = 3 # aperture size of the Sobel filter
    edges = cv.Canny(thresh1, lower_threshold, upper_threshold, aperture_size)

    #cv.imshow("edges", edges)
    #k = cv.waitKey(0)
    #cv.destroyWindow("edges")

    # noise reduction (part 1)
    blur1 = cv.fastNlMeansDenoising(edges, None, h=30, templateWindowSize=25, searchWindowSize=25)
    
    #cv.imshow("blur1", blur1)
    #k = cv.waitKey(0)
    #cv.destroyWindow("blur1")

    # dilation
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(blur1, element, iterations=14)

    #cv.imshow("dilated", dilated)
    #k = cv.waitKey(0)
    #cv.destroyWindow("dilated")

    # erosion
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    eroded = cv.erode(dilated, element, iterations=15)

    #cv.imshow("eroded", eroded)
    #k = cv.waitKey(0)
    #cv.destroyWindow("eroded")

    # noise reduction (part 2)
    blur2 = cv.fastNlMeansDenoising(eroded, None, h=25, templateWindowSize=25, searchWindowSize=25)
    
    #cv.imshow("blur2", blur2)
    #k = cv.waitKey(0)
    #cv.destroyWindow("blur2")

    # noise reduction (part 3)
    diameter = 30
    sigma_color = 100
    sigma_space = 30
    blur3 = cv.bilateralFilter(blur2, diameter, sigma_color, sigma_space)

    #cv.imshow("blur3", blur3)
    #k = cv.waitKey(0)
    #cv.destroyWindow("blur3")

    # thresholding (part 2)
    ret, thresh2 = cv.threshold(blur3, 150, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh2", thresh2)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh2")

    pre_processed_img = thresh2

    return pre_processed_img

def contour_detection(img, pre_processed_img):
    contours, hierarchy = cv.findContours(pre_processed_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    contour_img = img.copy()
    cv.drawContours(contour_img, contours, -1, (0, 255, 0), 5)

    #cv.imshow("contour_img", contour_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("contour_img")

    return contour_img, contours

def intersection_detection(contour_img, contours):
    centroids = []
    arrow_centroid = None

    height, width = contour_img.shape[:2]
    border_buffer = 25
    min_distance_to_border = float('inf')

    for contour in contours:
        M = cv.moments(contour)
        area = cv.contourArea(contour)
        if M['m00'] != 0 and 0 < area < 4000:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            distance_to_border = min(cx, cy, width - cx, height - cy)
            if distance_to_border > border_buffer: # avoid border contours caused by scanning
                if distance_to_border < min_distance_to_border and (((width/2 - 150) < cx < (width/2 + 150)) or ((height/2 - 150) < cy < (height/2 + 150))): # isolate arrow centroid
                    min_distance_to_border = distance_to_border
                    if arrow_centroid is not None:
                        centroids.append([arrow_centroid[0], arrow_centroid[1]])
                    arrow_centroid = cx, cy
                else:
                    centroids.append([cx, cy])

    # merge intersections
    merged_centroids = []
    centroids_array = np.array(centroids)
    clustering = DBSCAN(eps=120, min_samples=1).fit(centroids_array) 
    labels = clustering.labels_
    
    for label in set(labels):
        if label != -1: # ignore noise points
            cluster_points = centroids_array[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            merged_centroids.append(tuple(centroid))

    intersection_img = contour_img.copy()
    for centroid in merged_centroids:
        cv.circle(intersection_img, (int(centroid[0]), int(centroid[1])), 8, (0, 255, 255), -1)
    cv.circle(intersection_img, arrow_centroid, 8, (0, 0, 0), -1)

    #cv.imshow("intersection_img", intersection_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("intersection_img")
    
    return intersection_img, merged_centroids, arrow_centroid

def orient_image(intersection_img, merged_centroids, arrow_centroid):
    # determine side arrow is on assuming arrow is centred
    def get_closest_side(img, arrow_centroid):
        height, width = img.shape[:2]

        if arrow_centroid is not None:
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

    closest_side = get_closest_side(intersection_img, arrow_centroid)
    angle = rotation_based_on_side(closest_side)
    rotated_img, rotated_centroids = rotate_image_and_centroids(intersection_img, merged_centroids, angle)

    #cv.imshow("rotated_img", rotated_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("rotated_img")

    return rotated_img, rotated_centroids

def target_detection(rotated_img, rotated_centroids, LineC_T_C1):
    superposition_img = rotated_img.copy()
    for centroid in LineC_T_C1:
        cv.circle(superposition_img, (int(centroid[0]), int(centroid[1])), 8, (0, 0, 0), -1)
    
    #cv.imshow("superposition_img", superposition_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("superposition_img")

    # determine subset of LineC_T_C1 that are detected
    detected_centroids = []
    distance_threshold = 100 # required proximity of detected centroids to template centroids
    for template_centroid in LineC_T_C1:
        for centroid in rotated_centroids:
            distance = np.sqrt((centroid[0] - template_centroid[0])**2 + (centroid[1] - template_centroid[1])**2)
            if distance < distance_threshold:
                detected_centroids.append(tuple(template_centroid))

    detected_img = superposition_img.copy()
    for centroid in detected_centroids:
        cv.circle(detected_img, (int(centroid[0]), int(centroid[1])), 8, (127, 127, 127), -1)
    
    #cv.imshow("detected_img", detected_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("detected_img")

    return detected_img, detected_centroids

def post_processing(detected_img, rotated_centroids, detected_centroids, LineC_T_C1):    
    # determine number of lines crossed on left and right sides
    left_centroids = []
    right_centroids = []
    scoring_img = detected_img.copy()

    for centroid in rotated_centroids:
        height, width = detected_img.shape[:2]
        x, y = centroid
        if x <= (width/2 - 150):
            left_centroids.append(centroid)
            cv.circle(scoring_img, (int(centroid[0]), int(centroid[1])), 8, (255, 0, 0), -1)
        elif x >= (width/2 + 150):
            right_centroids.append(centroid)
            cv.circle(scoring_img, (int(centroid[0]), int(centroid[1])), 8, (0, 0, 255), -1)
    
    #cv.imshow("scoring_img", scoring_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("scoring_img")

    LineC_LS = len(left_centroids)
    LineC_RS = len(right_centroids)
    LineC = LineC_LS + LineC_RS

    # determine standard value
    mapping = {(0, 1): 0.0,
               (2, 3): 0.5,
               (4, 5): 1.0,
               (6,): 1.5,
               (7, 8): 2.0,
               (9, 10): 2.5,
               (11, 12): 3.0,
               (13, 14): 3.5,
               (15, 16): 4.0,
               (17,): 4.5,
               (18, 19): 5.0,
               (20,): 5.5,
               (21,): 6.0,
               (22, 23): 6.5,
               (24, 25): 7.0,
               (26, 27): 7.5,
               (28,): 8.0,
               (29, 30): 8.5,
               (31, 32): 9.0,
               (33, 34): 9.5,
               (35, 36): 10.0}

    LineC_SV = None
    for interval, standard_value in mapping.items():
        if len(interval) == 2:
            if interval[0] <= LineC <= interval[1]:
                LineC_SV = standard_value
                break
        elif len(interval) == 1:
            if interval[0] == LineC:
                LineC_SV = standard_value
                break
    
    # determine horizontal and vertical centres of cancellation
    def calculate_CoC(detected_centroids, LineC_T_C1):
        # calculate the mean positions
        mean_x_targets, mean_y_targets = np.mean(LineC_T_C1, axis=0)
        mean_x_detected, mean_y_detected = np.mean(detected_centroids, axis=0)

        # calculate the leftmost, bottommost, rightmost, and topmost targets
        leftmost_target, bottommost_target = np.min(LineC_T_C1, axis=0)
        rightmost_target, topmost_target = np.max(LineC_T_C1, axis=0)

        # adjust the scale so that range of targets is from -1 to 1 with the mean of targets being 0
        LineC_HCoC = round(2 * (mean_x_detected - mean_x_targets) / (rightmost_target - leftmost_target), 2)
        LineC_VCoC = round(2 * (mean_y_detected - mean_y_targets) / (topmost_target - bottommost_target), 2)

        return LineC_HCoC, LineC_VCoC

    LineC_HCoC, LineC_VCoC = calculate_CoC(detected_centroids, LineC_T_C1)

    return LineC_LS, LineC_RS, LineC, LineC_SV, LineC_HCoC, LineC_VCoC

def process_image(file_path, LineC_T_C1):
    img = image_acquisition(file_path)
    pre_processed_img = image_pre_processing(img)
    contour_img, contours = contour_detection(img, pre_processed_img)
    intersection_img, merged_centroids, arrow_centroid = intersection_detection(contour_img, contours)
    rotated_img, rotated_centroids = orient_image(intersection_img, merged_centroids, arrow_centroid)
    detected_img, detected_centroids = target_detection(rotated_img, rotated_centroids, LineC_T_C1)
    LineC_LS, LineC_RS, LineC, LineC_SV, LineC_HCoC, LineC_VCoC = post_processing(detected_img, rotated_centroids, detected_centroids, LineC_T_C1)
    return LineC_LS, LineC_RS, LineC, LineC_SV, LineC_HCoC, LineC_VCoC