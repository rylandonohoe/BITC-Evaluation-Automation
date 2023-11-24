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

    # resize the image to standard resolution 1650 x 2340
    resized = cv.resize(img, (1650, 2340))

    #cv.imshow("resized", resized)
    #k = cv.waitKey(0)
    #cv.destroyWindow("resized")

    return resized

def isolate_diamond(resized):
    # grayscale conversion
    gray1 = cv.cvtColor(resized, cv.COLOR_BGR2GRAY) # image is now 1-channel

    #cv.imshow("gray1", gray1)
    #k = cv.waitKey(0)
    #cv.destroyWindow("gray1")

    # corner detection
    blockSize = 40 # size of neighbourhood considered for corner detection
    kSize = 27 # aperture parameter of the Sobel derivative used
    k = 0.05 # Harris detector free parameter in the equation
    dst = cv.cornerHarris(gray1, blockSize, kSize, k)
    corners = np.where(dst > 0.02 * dst.max())
    
    corner_img = resized.copy()
    corner_img[corners] = [0, 255, 0]

    #cv.imshow("corner_img", corner_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("corner_img")

    # merge corners
    merged_corners = []
    corner_coordinates = np.array(list(zip(corners[1], corners[0]))) # format corners into list of (x, y) coordinates
    clustering = DBSCAN(eps=15, min_samples=200).fit(corner_coordinates)
    labels = clustering.labels_

    for label in set(labels):
        if label != -1: # ignore noise points
            cluster_points = corner_coordinates[labels == label]
            mean_corner = np.mean(cluster_points, axis=0)
            merged_corners.append(tuple(mean_corner))
    
    merged_img = resized.copy()
    for corner in merged_corners:
        cv.circle(merged_img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)

    #cv.imshow("merged_img", merged_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("merged_img")

    # isolate landmark corners
    landmark_corners = []

    height, width = resized.shape[:2]
    width_range = (int(14/32 * width), int(17/32 * width))
    height_range_1 = (int(10/32 * height), int(12/32 * height))
    height_range_2 = (int(20/32 * height), int(22/32 * height))

    for corner in merged_corners:
        if width_range[0] <= corner[0] <= width_range[1] and (height_range_1[0] <= corner[1] <= height_range_1[1] or height_range_2[0] <= corner[1] <= height_range_2[1]):
            landmark_corners.append(corner)

    if len(landmark_corners) != 2:
        raise Exception("Landmark corners not determined correctly.")
    
    landmark_img = resized.copy()
    for corner in landmark_corners:
        cv.circle(landmark_img, (int(corner[0]), int(corner[1])), 8, (0, 255, 0), -1)

    #cv.imshow("landmark_img", landmark_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("landmark_img")

    # determine rectangle points
    landmark1 = np.array(landmark_corners[0])
    landmark2 = np.array(landmark_corners[1])
    distance = np.linalg.norm(landmark1 - landmark2)
    direction = (landmark1 - landmark2) / distance
    perpendicular_direction = np.array([-direction[1], direction[0]])

    start_point1 = tuple(landmark1.astype(int))
    end_point1 = tuple((landmark1 + perpendicular_direction * (distance * 1.1)).astype(int))
    start_point2 = tuple(landmark2.astype(int))
    end_point2 = tuple((landmark2 + perpendicular_direction * (distance * 1.1)).astype(int))
    points = [start_point1, end_point1, start_point2, end_point2]

    point_img = resized.copy()
    for point in points:
        cv.circle(point_img, (point[0], point[1]), 8, (255, 0, 0), -1)
    
    #cv.imshow("point_img", point_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("point_img")

    # shrink rectangle
    np_points = np.array(points, dtype=np.float32)
    centre = np.mean(np_points, axis=0)
    shrink_factor = 0.05
    shrunken_points = []

    for point in np_points:
        direction = centre - point
        shrunken_point = point + shrink_factor * direction
        shrunken_points.append(tuple(shrunken_point.astype(int)))
    
    shrunken_point_img = resized.copy()
    for shrunken_point in shrunken_points:
        cv.circle(shrunken_point_img, (shrunken_point[0], shrunken_point[1]), 8, (255, 0, 0), -1)
    
    #cv.imshow("shrunken_point_img", shrunken_point_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("shrunken_point_img")

    # transform points and crop image
    new_width = int(np.linalg.norm(np.array(shrunken_points[0]) - np.array(shrunken_points[1])))
    new_height = int(np.linalg.norm(np.array(shrunken_points[0]) - np.array(shrunken_points[2])))
    dst_points = np.array([[0, 0], [new_width - 1, 0], [0, new_height - 1], [new_width - 1, new_height - 1]], dtype=np.float32)

    matrix = cv.getPerspectiveTransform(np.array(shrunken_points, dtype=np.float32), dst_points)
    cropped_img = cv.warpPerspective(resized, matrix, (new_width, new_height))

    #cv.imshow("cropped_img", cropped_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("cropped_img")

    return cropped_img

def image_pre_processing(cropped_img):
    # grayscale conversion
    gray1 = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY) # image is now 1-channel

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
    blur = cv.fastNlMeansDenoising(edges, None, h=30, templateWindowSize=30, searchWindowSize=25)
    
    #cv.imshow("blur", blur)
    #k = cv.waitKey(0)
    #cv.destroyWindow("blur")

    # thresholding
    ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh", thresh)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh")

    # dilation
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    dilated = cv.dilate(thresh, element, iterations=4)

    #cv.imshow("dilated", dilated)
    #k = cv.waitKey(0)
    #cv.destroyWindow("dilated")

    # erosion
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    eroded = cv.erode(dilated, element, iterations=4)

    #cv.imshow("eroded", eroded)
    #k = cv.waitKey(0)
    #cv.destroyWindow("eroded")

    pre_processed_img = eroded

    return pre_processed_img

def line_detection(cropped_img, pre_processed_img):
    # line detection
    rho = 1 # distance resolution
    theta = np.pi / 180 # angular resolution
    threshold = 10 # minimum number of intersections to detect a line
    min_line_length = 5 # minimum number of points to form a line
    max_line_gap = 5 # maximum gap between points to be considered in the same line
    lines = cv.HoughLinesP(pre_processed_img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    formatted_lines = [[[x1, y1], [x2, y2]] for [[x1, y1, x2, y2]] in lines]

    # draw lines on line_img
    line_img = cropped_img.copy()
    line_thickness = 5
    for line in formatted_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

    #cv.imshow("line_img", line_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("line_img")

    return line_img, formatted_lines

def line_processing(cropped_img, formatted_lines):
    # remove lines that are not part of the diamond
    filtered_lines = []
    
    # find the line closest to the center
    height, width = cropped_img.shape[:2]
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
    filtered_line_img = cropped_img.copy()
    for line in filtered_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        cv.line(filtered_line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    #cv.imshow("filtered_line_img", filtered_line_img)
    #cv.waitKey(0)
    #cv.destroyWindow("filtered_line_img")

    return filtered_line_img, filtered_lines

def ideal_diamond(filtered_line_img, filtered_lines):
    # find topmost and bottommost points of patient diamond
    top = bottom = filtered_lines[0][0]
    for line in filtered_lines:
        for point in line:
            x, y = point
            if y < top[1]:
                top = point
            if y > bottom[1]:
                bottom = point

    def find_extreme_point(filtered_lines, direction, roi):
        extreme_point = None
        extreme_x = float('inf') if direction == 'left' else float('-inf')

        for line in filtered_lines:
            for point in line:
                x, y = point
                if y < roi[0] or y > roi[1]: # check if point is within ROI
                    continue
                if (direction == 'left' and x < extreme_x) or (direction == 'right' and x > extreme_x):
                    extreme_point = point
                    extreme_x = x

        return extreme_point
    
    # find leftmost and rightmost points of patient diamond
    height = filtered_line_img.shape[0]
    initial_roi = (0, height)
    left = find_extreme_point(filtered_lines, 'left', initial_roi)
    right = find_extreme_point(filtered_lines, 'right', initial_roi)

    def interpolate_lines(filtered_lines, num_points):
        interpolated_lines = []
        for line in filtered_lines:
            x1, y1 = line[0]
            x2, y2 = line[1]
            xs = np.linspace(x1, x2, num_points).astype(int)
            ys = np.linspace(y1, y2, num_points).astype(int)
            interpolated_line = list(zip(xs, ys))
            interpolated_lines.extend(interpolated_line)
        return interpolated_lines
    
    # interpolate points within filtered lines
    points = interpolate_lines(filtered_lines, 500)

    def check_closedness(point, dense_points, x_threshold, y_threshold):
        x1, y1 = point
        above_closed = below_closed = False
        above_count = below_count = 0

        for (x2, y2) in dense_points:
            # check if point is different and within specified range in x direction
            if (x2, y2) != (x1, y1) and (x1 - x_threshold <= x2 <= x1 + x_threshold):
                if y1 - y_threshold <= y2 < y1: # check for points above
                    above_count += 1
                if y1 < y2 <= y1 + y_threshold: # check for points below
                    below_count += 1
        
        if above_count >= 100:
            above_closed = True
        if below_count >= 100:
            below_closed = True
        
        return above_closed, below_closed

    # check if left and right sides of patient diamond are closed above and below
    x_threshold = 30
    y_threshold = 5
    left_above_closed, left_below_closed = check_closedness(left, points, x_threshold, y_threshold)
    right_above_closed, right_below_closed = check_closedness(right, points, x_threshold, y_threshold)

    # find left_top and left_bottom points of patient diamond
    if (left_above_closed and left_below_closed) or not (left_above_closed or left_below_closed):
        left_top = left_bottom = left
    else:
        if left_above_closed:
            left_top = left
            left_bottom = find_extreme_point(filtered_lines, 'left', (left[1] + y_threshold, float('inf')))
        elif left_below_closed:
            left_bottom = left
            left_top = find_extreme_point(filtered_lines, 'left', (float('-inf'), left[1] - y_threshold))

    # find right_top and right_bottom points of patient diamond
    if (right_above_closed and right_below_closed) or not (right_above_closed or right_below_closed):
        right_top = right_bottom = right
    else:
        if right_above_closed:
            right_top = right
            right_bottom = find_extreme_point(filtered_lines, 'right', (right[1] + y_threshold, float('inf')))
        elif right_below_closed:
            right_bottom = right
            right_top = find_extreme_point(filtered_lines, 'right', (float('-inf'), right[1] - y_threshold))

    def line_intersection(line1, line2):
        # determinant of 2x2 matrix
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        
        delta_x = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]) # equivalent to (a, b)
        delta_y = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) # equivalent to (c, d)

        denominator = det(delta_x, delta_y)
        if denominator == 0:
            return None

        constants = (det(line1[0], line1[1]), det(line2[0], line2[1])) # equivalent to (e, f))
        x = int(det(constants, delta_x) / denominator)
        y = int(det(constants, delta_y) / denominator)
        
        return x, y
    
    # extrapolate intersection if necessary
    if left_top != left_bottom:
        left = line_intersection((top, left_top), (bottom, left_bottom))

    if right_top != right_bottom:
        right = line_intersection((top, right_top), (bottom, right_bottom))
    
    extreme_points = [top, right, bottom, left, left_top, left_bottom, right_top, right_bottom]

    # draw extreme_points
    extreme_points_img = filtered_line_img.copy()
    for extreme_point in extreme_points:
        cv.circle(extreme_points_img, extreme_point, 8, (0, 0, 255), -1)

    #cv.imshow("extreme_points_img", extreme_points_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("extreme_points_img")

    # construct diamond
    diamond_img = extreme_points_img.copy()
    colour = (255, 0, 0)
    cv.line(diamond_img, tuple(np.array(extreme_points[0]).astype(int)), tuple(np.array(extreme_points[1]).astype(int)), colour, 1) # top to right
    cv.line(diamond_img, tuple(np.array(extreme_points[1]).astype(int)), tuple(np.array(extreme_points[2]).astype(int)), colour, 1) # right to bottom
    cv.line(diamond_img, tuple(np.array(extreme_points[2]).astype(int)), tuple(np.array(extreme_points[3]).astype(int)), colour, 1) # bottom to left
    cv.line(diamond_img, tuple(np.array(extreme_points[3]).astype(int)), tuple(np.array(extreme_points[0]).astype(int)), colour, 1) # left to top
    cv.line(diamond_img, tuple(np.array(extreme_points[0]).astype(int)), tuple(np.array(extreme_points[2]).astype(int)), colour, 1) # top to bottom

    #cv.imshow("diamond_img", diamond_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("diamond_img")

    return diamond_img, extreme_points

def diamond_overlap(diamond_img, filtered_lines, extreme_points):
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
        point1 = [point[0] - dy*shift, point[1] + dx*shift]
        point2 = [point[0] + dy*shift, point[1] - dx*shift]

        return point1, point2

    # apply Cramer's rule to determine existence of unique line intersection from line endpoints
    def line_intersection(line1, line2):
        # determinant of 2x2 matrix
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        
        delta_x = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]) # equivalent to (a, b)
        delta_y = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) # equivalent to (c, d)

        denominator = det(delta_x, delta_y)
        if denominator != 0:
            constants = (det(line1[0], line1[1]), det(line2[0], line2[1])) # equivalent to (e, f))
            x = det(constants, delta_x) / denominator
            y = det(constants, delta_y) / denominator
        else:
            return None

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
    extremes = extreme_points[:4]
    ideal_lines = [interpolate_points(extremes[i], extremes[(i+1)%4], num_points) for i in range(4)] + [interpolate_points(extremes[0], extremes[2], num_points)]

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
                tolerance = 60
                perpendicular_line = calculate_perpendicular_line(point, dx, dy, tolerance)

                # check if any of the filtered_lines intersect the perpendicular line
                for line in filtered_lines:
                    intersection = line_intersection(line, perpendicular_line)
                    if intersection is not None:
                        valid_diamond_pixels[line_names[i]].append(point)
                        break

    # draw valid diamond pixels on diamond_img
    overlap_img = diamond_img.copy()
    for valid_line_pixels in valid_diamond_pixels.values():
        for pixel in valid_line_pixels:
            cv.circle(overlap_img, tuple(map(int, pixel)), 1, (0, 0, 255), -1)
    
    #cv.imshow("overlap_img", overlap_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("overlap_img")

    # determine valid lines
    validity_threshold = 0.75
    valid_lines = {key: len(value)/num_points >= validity_threshold for key, value in valid_diamond_pixels.items()}

    return overlap_img, valid_lines

def post_processing(extreme_points, valid_lines):
    top, right, bottom, left, left_top, left_bottom, right_top, right_bottom = extreme_points
    
    # ANGLES
    def calculate_angle(coordinates):
        A, B, C = coordinates

        # calculate lengths of sides of triangle
        AB = np.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)
        BC = np.sqrt((C[0] - B[0])**2 + (C[1] - B[1])**2)
        CA = np.sqrt((C[0] - A[0])**2 + (C[1] - A[1])**2)

        # apply the cosine law
        angle_radians = np.arccos((AB**2 + BC**2 - CA**2) / (2 * AB * BC))
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    # calculate angles
    CopyDiamond_TLA = round(calculate_angle([left, top, bottom]), 2) # top left angle
    CopyDiamond_TRA = round(calculate_angle([right, top, bottom]), 2) # top right angle
    CopyDiamond_BLA = round(calculate_angle([left, bottom, top]), 2) # bottom left angle
    CopyDiamond_BRA = round(calculate_angle([right, bottom, top]), 2) # bottom right angle
    CopyDiamond_AVA = round(calculate_angle([top, bottom, (bottom[0], top[1])]), 2) # absolute vertical angle
    CopyDiamond_AHA = round(calculate_angle([left, right, (left[0], right[1])]), 2) # absolute horzontal angle

    # AREAS
    def calculate_area(coordinates):
        area = 0
        
        for i in range(len(coordinates)):
            j = (i + 1) % len(coordinates)
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]
        area = abs(area) / 2.0

        return area
    
    # calculate area of left and right sides of patient diamond
    patient_left_area = calculate_area([top, left_top, left_bottom, bottom])
    patient_right_area = calculate_area([top, right_top, right_bottom, bottom])

    # calculate percentage of left and right sides of ideal diamond filled
    ideal_left_area = calculate_area([top, left, bottom])
    ideal_right_area = calculate_area([top, right, bottom])

    # calculate area ratios
    CopyDiamond_LAR = round(patient_left_area/ideal_left_area, 2) # left area ratio
    CopyDiamond_RAR = round(patient_right_area/ideal_right_area, 2) # right area ratio

    # LENGTHS
    def calculate_length(point1, point2):
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    # calculate lengths of patient and ideal diamond
    patient_top_left_length = calculate_length(top, left_top)
    ideal_top_left_length = calculate_length(top, left)
    patient_top_right_length = calculate_length(top, right_top)
    ideal_top_right_length = calculate_length(top, right)
    patient_bottom_left_length = calculate_length(bottom, left_bottom)
    ideal_bottom_left_length = calculate_length(bottom, left)
    patient_bottom_right_length = calculate_length(bottom, right_bottom)
    ideal_bottom_right_length = calculate_length(bottom, right)
    ideal_vertical_length = calculate_length(top, bottom)
    ideal_horizontal_length = calculate_length(left, right)

    # calculate length ratios
    CopyDiamond_TLLR = round(patient_top_left_length/ideal_top_left_length, 2) # top left length ratio
    CopyDiamond_TRLR = round(patient_top_right_length/ideal_top_right_length, 2) # top right length ratio
    CopyDiamond_BLLR = round(patient_bottom_left_length/ideal_bottom_left_length, 2) # bottom left length ratio
    CopyDiamond_BRLR = round(patient_bottom_right_length/ideal_bottom_right_length, 2) # bottom right length ratio
    CopyDiamond_VHLR = round(ideal_vertical_length/ideal_horizontal_length, 2) # vertical horizontal length ratio

    # SCORING
    # compute the shape score
    def shape_evaluation(valid_lines):
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
    def arrangement_evaluation(extreme_points):
        score = 0
        top, right, bottom, left, _, _, _, _ = extreme_points

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
    CopyDiamond_S = shape_evaluation(valid_lines)
    CopyDiamond_D = detail_evaluation(valid_lines)
    CopyDiamond_A = arrangement_evaluation(extreme_points)
    CopyDiamond = CopyDiamond_S + CopyDiamond_D + CopyDiamond_A

    # determine standard value
    mapping = {0: 0.0,
               1: 3.5,
               2: 6.5,
               3: 10.0}

    CopyDiamond_SV = None
    for score, standard_value in mapping.items():
        if CopyDiamond == score:
            CopyDiamond_SV = standard_value
            break
    
    return (CopyDiamond_S, CopyDiamond_D, CopyDiamond_A, CopyDiamond, CopyDiamond_SV,
            CopyDiamond_TLA, CopyDiamond_TRA, CopyDiamond_BLA, CopyDiamond_BRA, CopyDiamond_AVA, CopyDiamond_AHA,
            CopyDiamond_LAR, CopyDiamond_RAR,
            CopyDiamond_TLLR, CopyDiamond_TRLR, CopyDiamond_BLLR, CopyDiamond_BRLR, CopyDiamond_VHLR)

def process_image(file_path):
    resized = image_acquisition(file_path)
    cropped_img = isolate_diamond(resized)
    pre_processed_img = image_pre_processing(cropped_img)
    line_img, formatted_lines = line_detection(cropped_img, pre_processed_img)
    filtered_line_img, filtered_lines = line_processing(cropped_img, formatted_lines)
    diamond_img, extreme_points = ideal_diamond(filtered_line_img, filtered_lines)
    overlap_img, valid_lines = diamond_overlap(diamond_img, filtered_lines, extreme_points)
    (CopyDiamond_S, CopyDiamond_D, CopyDiamond_A, CopyDiamond, CopyDiamond_SV,
     CopyDiamond_TLA, CopyDiamond_TRA, CopyDiamond_BLA, CopyDiamond_BRA, CopyDiamond_AVA, CopyDiamond_AHA,
     CopyDiamond_LAR, CopyDiamond_RAR,
     CopyDiamond_TLLR, CopyDiamond_TRLR, CopyDiamond_BLLR, CopyDiamond_BRLR, CopyDiamond_VHLR) = post_processing(extreme_points, valid_lines)
    return (CopyDiamond_S, CopyDiamond_D, CopyDiamond_A, CopyDiamond, CopyDiamond_SV,
            CopyDiamond_TLA, CopyDiamond_TRA, CopyDiamond_BLA, CopyDiamond_BRA, CopyDiamond_AVA, CopyDiamond_AHA,
            CopyDiamond_LAR, CopyDiamond_RAR,
            CopyDiamond_TLLR, CopyDiamond_TRLR, CopyDiamond_BLLR, CopyDiamond_BRLR, CopyDiamond_VHLR)