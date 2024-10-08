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

def isolate_star(resized):
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

    if landmark1[1] > landmark2[1]:
        landmark1, landmark2 = landmark2, landmark1
    landmark2 = landmark2 + (2 * distance * direction)
    landmark1, landmark2 = landmark2, landmark1

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

    # thresholding
    ret, thresh = cv.threshold(gray1, 225, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh", thresh)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh")

    pre_processed_img = thresh
    
    return pre_processed_img

def corner_detection(cropped_img, pre_processed_img):
    # corner detection
    blockSize = 40 # size of neighbourhood considered for corner detection
    kSize = 27 # aperture parameter of the Sobel derivative used
    k = 0.05 # Harris detector free parameter in the equation
    dst = cv.cornerHarris(pre_processed_img, blockSize, kSize, k)
    corners = np.where(dst > 0.02 * dst.max())
    
    corner_img = cropped_img.copy()
    corner_img[corners] = [0, 255, 0]

    #cv.imshow("corner_img", corner_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("corner_img")
    
    return corner_img, corners

def corner_processing(cropped_img, corners):
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
    
    merged_img = cropped_img.copy()
    for corner in merged_corners:
        cv.circle(merged_img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
    
    #cv.imshow("merged_img", merged_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("merged_img")

    # remove outlier corners
    filtered_corners = []
    for i, corner1 in enumerate(merged_corners):
        if corner1 not in filtered_corners:
            for j, corner2 in enumerate(merged_corners):
                if i != j:
                    distance = np.sqrt((corner1[0] - corner2[0])**2 + (corner1[1] - corner2[1])**2)
                    if distance <= 400:
                        filtered_corners.append(corner1)
                        break

    # reduce filtered_corners count to 8 if necessary
    def merge_closest_corners(filtered_corners):
        min_distance = float('inf')
        min_pair = None
        for i, corner1 in enumerate(filtered_corners):
            for j, corner2 in enumerate(filtered_corners):
                if i != j:
                    distance = np.sqrt((corner1[0] - corner2[0])**2 + (corner1[1] - corner2[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        min_pair = (corner1, corner2)

        corner1, corner2 = min_pair
        mean_corner = ((corner1[0] + corner2[0]) / 2, (corner1[1] + corner2[1]) / 2)

        filtered_corners = [corner for corner in filtered_corners if corner not in min_pair]
        filtered_corners.append(mean_corner)

        return filtered_corners

    while len(filtered_corners) > 8:
        filtered_corners = merge_closest_corners(filtered_corners)

    filtered_img = cropped_img.copy()
    for corner in filtered_corners:
        cv.circle(filtered_img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
    
    #cv.imshow("filtered_img", filtered_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("filtered_img")
    
    # sort merged corners (clockwise starting from top)
    centre = np.mean(filtered_corners, axis=0)

    cartesian_polar = [] # convert Cartesian coordinates to polar coordinates to facilitate sorting
    for corner in filtered_corners:
        x, y = corner
        delta_x, delta_y = x - centre[0], y - centre[1]
        r = np.sqrt(delta_x**2 + delta_y**2)
        theta = np.arctan2(delta_x, -delta_y)
        if theta < 0:
            theta += 2 * np.pi
        cartesian_polar.append(((x, y), (r, theta)))

    cartesian_polar.sort(key=lambda coordinates: coordinates[1][1]) # sort corners by increasing theta value

    # sort starting at topmost corner
    topmost_corner_index = np.argmin([point[0][1] for point in cartesian_polar])
    cartesian_polar = cartesian_polar[topmost_corner_index:] + cartesian_polar[:topmost_corner_index]

    sorted_corners = [coordinates[0] for coordinates in cartesian_polar]

    sorted_img = cropped_img.copy()
    for corner in sorted_corners:
        cv.circle(sorted_img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
        
        #cv.imshow("sorted_img", sorted_img)
        #k = cv.waitKey(0)
        #cv.destroyWindow("sorted_img")

    return filtered_img, sorted_corners

def ideal_star(filtered_img, sorted_corners):
    # construct star
    star_img = filtered_img.copy()
    colour = (255, 0, 0)
    for i in range(len(sorted_corners)):
        cv.line(star_img, tuple(np.array(sorted_corners[i]).astype(int)), tuple(np.array(sorted_corners[(i+1) % len(sorted_corners)]).astype(int)), (255, 0, 0), 1)

    #cv.imshow("star_img", star_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("star_img")

    return star_img

def star_overlap(pre_processed_img, star_img, sorted_corners):
    # list points of each line in ideal star
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

    # create ideal lines
    num_lines = len(sorted_corners)
    num_points = 1000
    ideal_lines = [interpolate_points(sorted_corners[i], sorted_corners[(i+1)%num_lines], num_points) for i in range(num_lines)]

    # determine valid pixels on ideal star
    line_names = [f"Line{i}" for i in range(num_lines)]
    valid_star_pixels = {line_names[i]: [] for i in range(num_lines)}

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

                # find x- and y-values along perpendicular line
                x_values = np.linspace(perpendicular_line[0][0], perpendicular_line[1][0], num_points).astype(int)
                y_values = np.linspace(perpendicular_line[0][1], perpendicular_line[1][1], num_points).astype(int)

                # clip values to image dimensions
                x_values = np.clip(x_values, 0, star_img.shape[1] - 1)
                y_values = np.clip(y_values, 0, star_img.shape[0] - 1)

                # check if any black pixel is present on the perpendicular line
                mask = pre_processed_img == 0
                if np.any(mask[y_values, x_values]):
                    valid_star_pixels[f"Line{i}"].append(point)

    # draw valid star pixels on star_img
    overlap_img = star_img.copy()
    for valid_line_pixels in valid_star_pixels.values():
        for pixel in valid_line_pixels:
            cv.circle(overlap_img, tuple(map(int, pixel)), 1, (0, 0, 255), -1)
    
    #cv.imshow("overlap_img", overlap_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("overlap_img")

    # determine valid lines
    validity_threshold = 0.75
    valid_lines = {key: len(value)/num_points >= validity_threshold for key, value in valid_star_pixels.items()}

    return overlap_img, valid_lines

def post_processing(pre_processed_img, sorted_corners, valid_lines):
    # extract x and y values separately
    x_values = [point[0] for point in sorted_corners]
    y_values = [point[1] for point in sorted_corners]

    # find indices of the extreme points
    top_index = np.argmin(y_values)
    bottom_index = np.argmax(y_values)
    left_index = np.argmin(x_values)
    right_index = np.argmax(x_values)

    # find extreme points using their indices
    top = sorted_corners[top_index]
    bottom = sorted_corners[bottom_index]
    left = sorted_corners[left_index]
    right = sorted_corners[right_index]

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
    angles = []
    n = len(sorted_corners)
    for i in range(n):
        A = sorted_corners[i % n]
        B = sorted_corners[(i + 1) % n]
        C = sorted_corners[(i + 2) % n]
        angle = calculate_angle([A, B, C])
        angles.append(round(angle, 2))
    
    CopyStar_AIA = angles # all internal angles (clockwise from top-right)
    CopyStar_AVA = round(calculate_angle([top, bottom, (bottom[0], top[1])]), 2) # absolute vertical angle
    CopyStar_AHA = round(calculate_angle([left, right, (left[0], right[1])]), 2) # absolute horzontal angle

    # AREAS
    def calculate_area(coordinates):
        area = 0
        
        for i in range(len(coordinates)):
            j = (i + 1) % len(coordinates)
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]
        area = abs(area) / 2.0

        return area
    
    def get_cyclic_subset(start_index, end_index):
        if start_index <= end_index:
            return sorted_corners[start_index:end_index + 1]
        else:
            return sorted_corners[start_index:] + sorted_corners[:end_index + 1]
    
    # calculate areas
    CopyStar_LA = round(calculate_area(get_cyclic_subset(bottom_index, top_index)), 2) # left area
    CopyStar_RA = round(calculate_area(get_cyclic_subset(top_index, bottom_index)), 2) # right area
    CopyStar_TA = round(calculate_area(get_cyclic_subset(left_index, right_index)), 2) # top area
    CopyStar_BA = round(calculate_area(get_cyclic_subset(right_index, left_index)), 2) # bottom area

    # LENGTHS
    def calculate_length(point1, point2):
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    # calculate lengths
    lengths = []
    n = len(sorted_corners)
    for i in range(n):
        A = sorted_corners[i % n]
        B = sorted_corners[(i + 1) % n]
        length = calculate_length(A, B)
        lengths.append(round(length, 2))

    CopyStar_AL = lengths # all lengths (clockwise from top-right)

    # SCORING
    # compute the shape score
    def shape_evaluation(sorted_corners, valid_lines):
        score = 0

        # return immediately if star does not have an even number of corners to avoid indexing errors
        if len(sorted_corners) % 2 != 0:
            return score

        # find the primary and intermediate corners
        y_values = [point[1] for point in sorted_corners]
        topmost_index = np.argmin(y_values)
        primary_corner_indices = [(topmost_index + i) % len(sorted_corners) for i in range(0, len(sorted_corners), 2)]
        primary_corners = [sorted_corners[i] for i in primary_corner_indices]
        intermediate_corner_indices = [i for i in range(len(sorted_corners)) if i not in primary_corner_indices]
        intermediate_corners = [sorted_corners[i] for i in intermediate_corner_indices]

        # analyze radial distance of corners
        centre = np.mean(sorted_corners, axis=0)

        primary_corner_distances = [np.sqrt((primary_corner[0] - centre[0])**2 + (primary_corner[1] - centre[1])**2) for primary_corner in primary_corners]
        mean_primary_corner_distance = sum(primary_corner_distances) / len(primary_corner_distances)
        primary_corner_distance_differences = [abs(primary_corner_distance - mean_primary_corner_distance) for primary_corner_distance in primary_corner_distances]

        intermediate_corner_distances = [np.sqrt((intermediate_corner[0] - centre[0])**2 + (intermediate_corner[1] - centre[1])**2) for intermediate_corner in intermediate_corners]
        mean_intermediate_corner_distance = sum(intermediate_corner_distances) / len(intermediate_corner_distances)
        intermediate_corner_distance_differences = [abs(intermediate_corner_distance - mean_intermediate_corner_distance) for intermediate_corner_distance in intermediate_corner_distances]
        
        # determine top_bottom_distance
        topmost = sorted_corners[topmost_index]
        bottommost = sorted_corners[np.argmax(y_values)]
        top_bottom_distance = np.sqrt((bottommost[0] - topmost[0])**2 + (bottommost[1] - topmost[1])**2)
        
        if (all(valid_lines.values()) and
            max(primary_corner_distance_differences) <= 100 and
            max(intermediate_corner_distance_differences) <= 100 and
            25 <= (mean_primary_corner_distance - mean_intermediate_corner_distance) <= (top_bottom_distance - 25)):
            score = 1
        return score

    # compute the detail score
    def detail_evaluation(sorted_corners):
        score = 0
        if CopyStar_S == 1 and len(sorted_corners) == 8:
            score = 1
        return score
    
    # compute the arrangement score
    def arrangement_evaluation(sorted_corners):
        score = 0

        # return immediately if star is not 4-pointed to avoid indexing errors
        if CopyStar_D == 0:
            return score

        x_values = [point[0] for point in sorted_corners]
        y_values = [point[1] for point in sorted_corners]
        
        # determine the topmost, leftmost, bottommost, and rightmost corners
        topmost = sorted_corners[np.argmin(y_values)]
        bottommost = sorted_corners[np.argmax(y_values)]
        leftmost = sorted_corners[np.argmin(x_values)]
        rightmost = sorted_corners[np.argmax(x_values)]
        
        # calculate distances and differences
        top_bottom_distance = np.sqrt((bottommost[0] - topmost[0])**2 + (bottommost[1] - topmost[1])**2)
        left_right_distance = np.sqrt((rightmost[0] - leftmost[0])**2 + (rightmost[1] - leftmost[1])**2)
        
        delta_x_top_bottom = abs(topmost[0] - bottommost[0])
        delta_y_left_right = abs(leftmost[1] - rightmost[1])
        
        # find the intermediate corners
        topmost_index = np.argmin(y_values)
        intermediate_corner_indices = [(topmost_index + i) % 8 for i in [1, 3, 5, 7]]
        intermediate_corners = [sorted_corners[i] for i in intermediate_corner_indices]
        
        # calculate the distances and angles
        intermediate_distances = [np.sqrt((intermediate_corners[i][0] - intermediate_corners[(i+1)%4][0])**2 + (intermediate_corners[i][1] - intermediate_corners[(i+1)%4][1])**2) for i in range(4)]
        mean_distance = sum(intermediate_distances) / 4
        intermediate_distance_differences = [abs(intermediate_distance - mean_distance) for intermediate_distance in intermediate_distances]

        def calculate_angle(A, B, C):
            BA = np.array(A) - np.array(B)
            BC = np.array(C) - np.array(B)
            cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)

        intermediate_angles = [calculate_angle(intermediate_corners[(i-1)%4], intermediate_corners[i], intermediate_corners[(i+1)%4]) for i in range(4)]
        intermediate_angle_differences = [abs(angle - 90) for angle in intermediate_angles]
        
        if (abs(top_bottom_distance - left_right_distance) <= 200 and
            delta_x_top_bottom <= 200 and
            delta_y_left_right <= 200 and
            max(intermediate_distance_differences) <= 100 and
            max(intermediate_angle_differences) <= 30):
            score = 1
        return score

    # compute final score
    CopyStar_S = shape_evaluation(sorted_corners, valid_lines)
    CopyStar_D = detail_evaluation(sorted_corners)
    CopyStar_A = arrangement_evaluation(sorted_corners)
    CopyStar = CopyStar_S + CopyStar_D + CopyStar_A

    # determine standard value
    mapping = {0: 0.0,
               1: 3.5,
               2: 6.5,
               3: 10.0}

    CopyStar_SV = None
    for score, standard_value in mapping.items():
        if CopyStar == score:
            CopyStar_SV = standard_value
            break

    return (CopyStar_S, CopyStar_D, CopyStar_A, CopyStar, CopyStar_SV,
            CopyStar_AIA, CopyStar_AVA, CopyStar_AHA,
            CopyStar_LA, CopyStar_RA, CopyStar_TA, CopyStar_BA,
            CopyStar_AL)

def process_image(file_path):
    resized = image_acquisition(file_path)
    cropped_img = isolate_star(resized)
    pre_processed_img = image_pre_processing(cropped_img)
    corner_img, corners = corner_detection(cropped_img, pre_processed_img)
    filtered_img, sorted_corners = corner_processing(cropped_img, corners)
    star_img = ideal_star(filtered_img, sorted_corners)
    overlap_img, valid_lines = star_overlap(pre_processed_img, star_img, sorted_corners)
    (CopyStar_S, CopyStar_D, CopyStar_A, CopyStar, CopyStar_SV,
    CopyStar_AIA, CopyStar_AVA, CopyStar_AHA,
    CopyStar_LA, CopyStar_RA, CopyStar_TA, CopyStar_BA,
    CopyStar_AL) = post_processing(pre_processed_img, sorted_corners, valid_lines)
    return (CopyStar_S, CopyStar_D, CopyStar_A, CopyStar, CopyStar_SV,
            CopyStar_AIA, CopyStar_AVA, CopyStar_AHA,
            CopyStar_LA, CopyStar_RA, CopyStar_TA, CopyStar_BA,
            CopyStar_AL)