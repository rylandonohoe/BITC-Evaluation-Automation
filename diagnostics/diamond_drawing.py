import cv2 as cv
import numpy as np
import sys

def image_acquisition(file_path):
    img = cv.imread(cv.samples.findFile(file_path))

    if img is None:
        sys.exit("Could not read the image.") # make sure image is png, jpg, or jpeg (some other file types could work as well)

    #cv.imshow("img", img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("img")

    # isolate diamond
    height, width = img.shape[:2]
    resized = img[int(height/3 + 75):int(height/3*2 - 75), int(width/2 + 15):int(width - 45)]

    #cv.imshow("resized", resized)
    #k = cv.waitKey(0)
    #cv.destroyWindow("resized")

    return resized

def image_pre_processing(resized):
    # grayscale conversion
    gray1 = cv.cvtColor(resized, cv.COLOR_BGR2GRAY) # image is now 1-channel

    #cv.imshow("gray1", gray1)
    #k = cv.waitKey(0)
    #cv.destroyWindow("gray1")

    # edge detection
    lower_threshold = 50 # lower threshold value in Hysteresis Thresholding
    upper_threshold = 150 # upper threshold value in Hysteresis Thresholding
    aperture_size = 3 # aperture size of the Sobel filter
    edges = cv.Canny(gray1, lower_threshold, upper_threshold, aperture_size)

    #cv.imshow("edges", edges)
    #k = cv.waitKey(0)
    #cv.destroyWindow("edges")

    # noise reduction
    blur = cv.fastNlMeansDenoising(edges, None, h=25, templateWindowSize=25, searchWindowSize=25)
    
    #cv.imshow("blur", blur)
    #k = cv.waitKey(0)
    #cv.destroyWindow("blur")

    # thresholding
    ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh", thresh)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh")

    # dilation
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(thresh, element, iterations=4)

    #cv.imshow("dilated", dilated)
    #k = cv.waitKey(0)
    #cv.destroyWindow("dilated")

    # erosion
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    eroded = cv.erode(dilated, element, iterations=4)

    #cv.imshow("eroded", eroded)
    #k = cv.waitKey(0)
    #cv.destroyWindow("eroded")

    pre_processed_img = eroded

    return pre_processed_img

def line_detection(resized, pre_processed_img):
    # line detection
    rho = 1 # distance resolution
    theta = np.pi / 180 # angular resolution
    threshold = 10 # minimum number of intersections to detect a line
    min_line_length = 5 # minimum number of points to form a line
    max_line_gap = 5 # maximum gap between points to be considered in the same line
    lines = cv.HoughLinesP(pre_processed_img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    formatted_lines = [[[x1, y1], [x2, y2]] for [[x1, y1, x2, y2]] in lines]

    # draw lines on line_img
    line_img = resized.copy()
    line_thickness = 5
    for line in formatted_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

    #cv.imshow("line_img", line_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("line_img")

    return line_img, formatted_lines

def line_processing(resized, formatted_lines):
    # remove lines that are not part of the diamond
    filtered_lines = []
    
    # find the line closest to the center
    height, width = resized.shape[:2]
    centre = [width // 2, height // 2]
    min_distance = float('inf')
    min_index = None
    for i, line in enumerate(formatted_lines):
        for point in line:
            distance = np.sqrt((point[0] - centre[0])**2 + (point[1] - centre[1])**2)
            if distance < min_distance:
                min_distance = distance
                min_index = i
    filtered_lines.append(formatted_lines[min_index])

    # append to filtered_lines any lines within line_threshold of any line in filtered_lines
    line_threshold = 100
    any_added = True
    while any_added:
        any_added = False
        for line in formatted_lines:
            if line not in filtered_lines:
                for filtered_line in filtered_lines:
                    distances = [np.sqrt((x1 - x2)**2 + (y1 - y2)**2) for x1, y1 in line for x2, y2 in filtered_line]
                    if any(distance <= line_threshold for distance in distances):
                        filtered_lines.append(line)
                        any_added = True
                        break

    # draw lines on filtered_line_img
    filtered_line_img = resized.copy()
    line_thickness = 5
    for line in filtered_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        cv.line(filtered_line_img, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

    #cv.imshow("filtered_line_img", filtered_line_img)
    #cv.waitKey(0)
    #cv.destroyWindow("filtered_line_img")

    return filtered_line_img, filtered_lines

def ideal_diamond(line_img, formatted_lines):
    # find corners
    top = formatted_lines[0][0]
    right = formatted_lines[0][0]
    bottom = formatted_lines[0][0]
    left = formatted_lines[0][0]

    for line in formatted_lines:
        for point in line:
            x, y = point
            if y > top[1]:
                top = point
            if x > right[0]:
                right = point
            if y < bottom[1]:
                bottom = point
            if x < left[0]:
                left = point

    corners = [top, right, bottom, left]

    # draw corners
    corner_img = line_img.copy()
    corner_thickness = 8
    for corner in corners:
        cv.circle(corner_img, corner, corner_thickness, (0, 0, 255), -1)

    #cv.imshow("corner_img", corner_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("corner_img")

    # construct diamond
    diamond_img = corner_img.copy()
    line_thickness = 1
    colour = (255, 0, 0)
    cv.line(diamond_img, tuple(np.array(corners[0]).astype(int)), tuple(np.array(corners[1]).astype(int)), colour, line_thickness) # top to right
    cv.line(diamond_img, tuple(np.array(corners[1]).astype(int)), tuple(np.array(corners[2]).astype(int)), colour, line_thickness) # right to bottom
    cv.line(diamond_img, tuple(np.array(corners[2]).astype(int)), tuple(np.array(corners[3]).astype(int)), colour, line_thickness) # bottom to left
    cv.line(diamond_img, tuple(np.array(corners[3]).astype(int)), tuple(np.array(corners[0]).astype(int)), colour, line_thickness) # left to top
    cv.line(diamond_img, tuple(np.array(corners[0]).astype(int)), tuple(np.array(corners[2]).astype(int)), colour, line_thickness) # top to bottom

    #cv.imshow("diamond_img", diamond_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("diamond_img")

    return diamond_img, corners

def diamond_overlap(diamond_img, formatted_lines, corners):
    # list points of each line in ideal diamond
    def interpolate_points(point1, point2, num_points):
        # interpolate between two points
        x_values = np.linspace(point1[0], point2[0], num_points)
        y_values = np.linspace(point1[1], point2[1], num_points)
        return list(zip(x_values, y_values))
    
    # define perpendicular line to the direction given by dx and dy
    def calculate_perpendicular_line(point, dx, dy, tolerance):
        shift = tolerance / np.sqrt(2)

        # compute the points of the perpendicular line
        p1 = [point[0] - dy*shift, point[1] + dx*shift]
        p2 = [point[0] + dy*shift, point[1] - dx*shift]

        return p1, p2

    # apply Cramer's rule to determine existence of unique line intersection from line endpoints
    def line_intersection(line1, line2):
        # determinant of 2x2 matrix
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        
        delta_x = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]) # equivalent to (a, b)
        delta_y = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) # equivalent to (c, d)

        denominator = det(delta_x, delta_y)
        if denominator == 0:
            return None

        constants = (det(*line1), det(*line2)) # equivalent to (e, f))
        x = det(constants, delta_x) / denominator
        y = det(constants, delta_y) / denominator
        
        # check if the intersection point lies within both line segments
        if (min(line1[0][0], line1[1][0]) <= x <= max(line1[0][0], line1[1][0]) and
            min(line1[0][1], line1[1][1]) <= y <= max(line1[0][1], line1[1][1]) and
            min(line2[0][0], line2[1][0]) <= x <= max(line2[0][0], line2[1][0]) and
            min(line2[0][1], line2[1][1]) <= y <= max(line2[0][1], line2[1][1])):
            return x, y
        else:

            return None

    # create ideal lines
    num_points = 1000
    ideal_lines = [interpolate_points(corners[i], corners[(i+1)%4], num_points) for i in range(4)] + [interpolate_points(corners[0], corners[2], num_points)]

    # determine valid pixels on ideal diamond
    line_names = ["top_right", "right_bottom", "bottom_left", "left_top", "top_bottom"]
    valid_diamond_pixels = {line_names[i]: [] for i in range(5)}
    
    for i, ideal_line in enumerate(ideal_lines):
        # calculate dx and dy of ideal line
        delta_x, delta_y = ideal_line[1][0] - ideal_line[0][0], ideal_line[1][1] - ideal_line[0][1]
        denominator = np.sqrt(delta_x**2 + delta_y**2)
        
        if denominator != 0:
            dx = delta_x / denominator
            dy = delta_y / denominator
            
            for point in ideal_line:
                # calculate perpendicular line
                tolerance = 100
                perpendicular_line = calculate_perpendicular_line(point, dx, dy, tolerance)

                # check if any of the formatted_lines intersect the perpendicular line
                for formatted_line in formatted_lines:
                    intersection = line_intersection(formatted_line, perpendicular_line)
                    if intersection is not None:
                        valid_diamond_pixels[line_names[i]].append(point)
                        break

    # draw valid diamond pixels on diamond_img
    overlap_img = diamond_img.copy()
    pixel_thickness = 1
    for valid_line_pixels in valid_diamond_pixels.values():
        for pixel in valid_line_pixels:
            cv.circle(overlap_img, tuple(map(int, pixel)), pixel_thickness, (0, 0, 255), -1)
    
    #cv.imshow("overlap_img", overlap_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("overlap_img")

    # determine valid lines
    validity_threshold = 0.8
    valid_lines = {key: len(value)/num_points >= validity_threshold for key, value in valid_diamond_pixels.items()}

    return overlap_img, valid_lines

def post_processing(corners, valid_lines):
    # compute the form score
    def form_evaluation(valid_lines):
        score = 0
        if all(valid_lines[line] for line in ["top_right", "right_bottom", "bottom_left", "left_top"]):
            score = 1
        return score

    # compute the detail score
    def detail_evaluation(valid_lines):
        score = 0
        if valid_lines["top_bottom"]:
            score = 1
        return score
        
    # compute the arrangement score
    def arrangement_evaluation(corners):
        score = 0
        top, right, bottom, left = corners

        top_bottom_distance = np.sqrt((bottom[0] - top[0]) ** 2 + (bottom[1] - top[1]) ** 2)
        left_right_distance = np.sqrt((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2)
        
        delta_x_top_bottom = abs(top[0] - bottom[0])
        delta_y_left_right = abs(left[1] - right[1])

        if (abs(top_bottom_distance - left_right_distance) <= 200 and
            delta_x_top_bottom <= 200 and
            delta_y_left_right <= 200):
            score = 1
        return score
    
    # compute final score
    DrawDiamond_F = form_evaluation(valid_lines)
    DrawDiamond_D = detail_evaluation(valid_lines)
    DrawDiamond_A = arrangement_evaluation(corners)
    DrawDiamond = DrawDiamond_F + DrawDiamond_D + DrawDiamond_A

    # determine standard value
    mapping = {0: 0.0,
               1: 3.5,
               2: 6.5,
               3: 10.0}

    DrawDiamond_SV = None
    for score, standard_value in mapping.items():
        if DrawDiamond == score:
            DrawDiamond_SV = standard_value
            break
    
    return DrawDiamond_F, DrawDiamond_D, DrawDiamond_A, DrawDiamond, DrawDiamond_SV

def process_image(file_path):
    resized = image_acquisition(file_path)
    pre_processed_img = image_pre_processing(resized)
    line_img, formatted_lines = line_detection(resized, pre_processed_img)
    filtered_line_img, filtered_lines = line_processing(resized, formatted_lines)
    diamond_img, corners = ideal_diamond(filtered_line_img, filtered_lines)
    overlap_img, valid_lines = diamond_overlap(diamond_img, filtered_lines, corners)
    DrawDiamond_F, DrawDiamond_D, DrawDiamond_A, DrawDiamond, DrawDiamond_SV = post_processing(corners, valid_lines)
    return DrawDiamond_F, DrawDiamond_D, DrawDiamond_A, DrawDiamond, DrawDiamond_SV

#for name in ["Braun", "BW", "Daskalon", "Dotzamer", "Franz", "Gerke", "Kuhn", "Loffelad", "Sigruner"]:
    #file_path = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/patients/" + name + "/Draw.png"
    #print(process_image(file_path))