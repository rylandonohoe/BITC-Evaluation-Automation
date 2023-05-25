import cv2 as cv
import math
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
    # edge enhancement (unsharp masking)
    blurred = cv.GaussianBlur(img, (0, 0), 3)
    sharpened = cv.addWeighted(img, 1.5, blurred, -0.5, 0)

    # contrast adjustment (histogram equalization)
    gray1 = cv.cvtColor(sharpened, cv.COLOR_BGR2GRAY)
    equalized_gray1 = cv.equalizeHist(gray1)
    gray3 = cv.cvtColor(equalized_gray1, cv.COLOR_GRAY2BGR) # convert gray1 to three channel for addWeighted function

    # noise reduction (Gaussian blur)
    kernel_size = (7, 7) # larger = blurrier
    blur_gray3 = cv.GaussianBlur(gray3, kernel_size, 0)
    
    #cv.imshow("Display window", blur_gray3)
    #k = cv.waitKey(0)

    return blur_gray3

def edge_and_line_detection(blur_gray3):
    line_img = np.zeros((blur_gray3.shape[0], blur_gray3.shape[1], 3), dtype=np.uint8) # blank image with same size as blur_gray3
    line_img[:] = (255, 255, 255) # background of line_img set to white

    # edge detection
    lower_threshold = 25 # lower threshold value in Hysteresis Thresholding
    upper_threshold = 100 # upper threshold value in Hysteresis Thresholding
    aperture_size = 7 # aperture size of the Sobel filter
    edges = cv.Canny(blur_gray3, lower_threshold, upper_threshold, aperture_size)

    # line detection
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi / 180 # angular resolution in radians of the Hough grid
    threshold = 5 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 # minimum number of pixels making up a line
    max_line_gap = 15 # maximum gap in pixels between connectable line segments
    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    formatted_lines = [[[x1, y1], [x2, y2]] for [[x1, y1, x2, y2]] in lines]

    # draw the lines on line_img
    line_thickness = 5
    for line in formatted_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        cv.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), line_thickness)

    line_blur_gray3 = cv.addWeighted(blur_gray3, 0.8, line_img, 1, 0)

    #cv.imshow("Display window", line_blur_gray3)
    #k = cv.waitKey(0)

    return line_blur_gray3, formatted_lines

def intersection_detection(line_blur_gray3, formatted_lines):
    # intersection detection
    def lines_similar(line1, line2, slope_threshold):
        slope1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0]) if line1[1][0] != line1[0][0] else 1000000
        slope2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0]) if line2[1][0] != line2[0][0] else 1000000
        
        return abs(slope1 - slope2) < slope_threshold # true if slopes of lines are similar enough

    def det(a, b, c, d):
        return a * d - b * c

    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
        x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]

        det1_and_2 = det(x1, y1, x2, y2)
        det3_and_4 = det(x3, y3, x4, y4)
        denominator = det(x1 - x2, y1 - y2, x3 - x4, y3 - y4)

        if np.isclose(denominator, 0):
            return None
        
        x = (det(det1_and_2, x1 - x2, det3_and_4, x3 - x4) / denominator)
        y = (det(det1_and_2, y1 - y2, det3_and_4, y3 - y4) / denominator)

        # check if intersection point falls within the domain and range of both lines and is not along the image border
        border_threshold = 130 # how far the arrow is from the border of the page

        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) and min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4):
            if border_threshold <= x <= line_blur_gray3.shape[1] - border_threshold and border_threshold <= y <= line_blur_gray3.shape[0] - border_threshold: 
                return x, y
        else:
            return None

    intersections = []
    for i in range(len(formatted_lines)):
        for j in range(i+1, len(formatted_lines)):
            if not lines_similar(formatted_lines[i], formatted_lines[j], 1.0): # 1.0 as slope threshold
                intersection = line_intersection(formatted_lines[i], formatted_lines[j])
                if intersection:
                    intersections.append(intersection)

    # merge intersections
    def dbscan_merge_points(points, epsilon):
        points_array = np.array(points)
        clustering = DBSCAN(eps=epsilon, min_samples=1).fit(points_array)
        labels = clustering.labels_

        merged_points = []

        for label in set(labels):
            cluster_points = points_array[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            merged_points.append(tuple(centroid))

        return merged_points

    cluster_threshold = 50
    merged_intersections = dbscan_merge_points(intersections, cluster_threshold)

    # draw the merged intersections on line_blur_gray3
    intersection_line_blur_gray3 = line_blur_gray3
    intersection_thickness = 8
    for intersection in merged_intersections:
        cv.circle(intersection_line_blur_gray3, tuple(map(int, intersection)), intersection_thickness, (0, 255, 0), -1)

    #cv.imshow("Display window", intersection_line_blur_gray3)
    #k = cv.waitKey(0)

    return intersection_line_blur_gray3, merged_intersections

def post_processing(intersection_line_blur_gray3, merged_intersections):
    # determine standard value
    LineC = len(merged_intersections) - 4 # subtract the 4 crosses in the middle

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

    # orienting image
    def find_first_nonwhite_pixel(img):
        noise_threshold = 100
        sides = [("top", img[noise_threshold:]), 
                ("right", np.rot90(img[:, :-noise_threshold])), 
                ("bottom", np.flipud(img[:-noise_threshold])), 
                ("left", np.rot90(img[:, noise_threshold:]))]
        nonwhite_coords = {}
        for side, side_img in sides:
            nonwhite_coords[side] = next(((i+noise_threshold,j) for i,row in enumerate(side_img) for j,pixel in enumerate(row) if np.any(pixel != 255)), None)
        return nonwhite_coords

    def rotation_based_on_side(side):
        return {"top": 180, "right": 90, "bottom": 0, "left": 270}.get(side, 0)

    def rotate_image(img, angle):
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)

        M = cv.getRotationMatrix2D(center, angle, 1.0)

        # calculate new image size after rotation
        r = np.deg2rad(angle)
        new_w = int(abs(h * np.sin(r)) + abs(w * np.cos(r)))
        new_h = int(abs(h * np.cos(r)) + abs(w * np.sin(r)))

        M[0, 2] += new_w / 2 - center[0]
        M[1, 2] += new_h / 2 - center[1]

        rotated = cv.warpAffine(img, M, (new_w, new_h))

        return rotated

    # find the side with the arrow and rotate the image
    nonwhite_coords = find_first_nonwhite_pixel(intersection_line_blur_gray3)
    closest_side = min(nonwhite_coords, key=nonwhite_coords.get)
    angle = rotation_based_on_side(closest_side)
    rotated_intersection_line_blur_gray3 = rotate_image(intersection_line_blur_gray3, angle)

    cv.imshow("Display window", rotated_intersection_line_blur_gray3)
    k = cv.waitKey(0)

    return LineC, LineC_SV



def process_image(file_path):
    img = image_acquisition(file_path)
    blur_gray3 = image_pre_processing(img)
    line_blur_gray3, formatted_lines = edge_and_line_detection(blur_gray3)
    intersection_line_blur_gray3, merged_intersections = intersection_detection(line_blur_gray3, formatted_lines)
    LineC, LineC_SV = post_processing(intersection_line_blur_gray3, merged_intersections)
    return LineC, LineC_SV

a = process_image("/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/patients/Daskalon/LineC.png")
print(a)