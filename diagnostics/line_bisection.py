import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import sys

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
    lower_threshold = 10 # lower threshold value in Hysteresis Thresholding
    upper_threshold = 20 # upper threshold value in Hysteresis Thresholding
    aperture_size = 3 # aperture size of the Sobel filter
    edges1 = cv.Canny(thresh1, lower_threshold, upper_threshold, aperture_size)

    #cv.imshow("Display window", edges1)
    #k = cv.waitKey(0)

    # noise reduction
    blur = cv.fastNlMeansDenoising(edges1, None, h=30, templateWindowSize=20, searchWindowSize=20)
    
    #cv.imshow("Display window", blur)
    #k = cv.waitKey(0)

    # dilation (part 1)
    element = cv.getStructuringElement(cv.MORPH_RECT, (33, 33))
    dilated1 = cv.dilate(blur, element, iterations=1)

    #cv.imshow("Display window", dilated1)
    #k = cv.waitKey(0)

    # thresholding (part 2)
    ret, thresh2 = cv.threshold(dilated1, 215, 255, cv.THRESH_BINARY)

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
    contours, hierarchy = cv.findContours(pre_processed_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    contour_img = img.copy()
    cv.drawContours(contour_img, contours, -1, (0, 255, 0), 5)

    #cv.imshow("Display window", contour_img)
    #k = cv.waitKey(0)

    return contour_img, contours

def contour_processing(img, contours):
    # compute centroids
    centroids = []
    for contour in contours:
        M = cv.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))

    # merge contours based on centroid proximity
    merged_contours = []
    distance_threshold = 200
    processed = np.zeros(len(contours), dtype=bool) # boolean array to keep track of which contours have already been placed in a group
    for i, centroid in enumerate(centroids):
        if not processed[i]:
            distances = np.array([np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2) for cx, cy in centroids])
            group = np.where(distances < distance_threshold)[0]
            processed[group] = True
            merged_contour = np.vstack([contours[j] for j in group]) # append all members of group into single contour
            merged_contours.append(merged_contour)

    # filter contours to remove noise contours
    filtered_contours = []
    arrow_contour = None

    height, width = img.shape[:2]
    border_buffer = 25
    min_distance_to_border = float('inf')
    for contour in merged_contours:
        M = cv.moments(contour)
        area = cv.contourArea(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            distance_to_border = min(cx, cy, width - cx, height - cy)
            if distance_to_border > border_buffer: # avoid border contours caused by scanning
                if distance_to_border < min_distance_to_border and area < 5000: # isolate arrow contour
                    min_distance_to_border = distance_to_border
                    if arrow_contour is not None:
                        filtered_contours.append(arrow_contour)
                    arrow_contour = contour
                else:
                    if distance_to_border > 5 * border_buffer and area > 5000: # isolate line contours
                        filtered_contours.append(contour)

    filtered_contour_img = img.copy()
    cv.drawContours(filtered_contour_img, filtered_contours, -1, (0, 255, 0), 5)
    cv.drawContours(filtered_contour_img, arrow_contour, -1, (0, 0, 0), 5)
    
    #cv.imshow("Display window", filtered_contour_img)
    #k = cv.waitKey(0)

    # reduce contour shape complexity
    processed_contours = []
    for contour in filtered_contours:
        epsilon = 0.0008 * cv.arcLength(contour, True)
        approx_contour = cv.approxPolyDP(contour, epsilon, True)
        processed_contours.append(approx_contour)

    processed_contour_img = img.copy()
    cv.drawContours(processed_contour_img, processed_contours, -1, (255, 0, 0), 5)
    cv.drawContours(processed_contour_img, arrow_contour, -1, (0, 0, 0), 5)
    
    #cv.imshow("Display window", processed_contour_img)
    #k = cv.waitKey(0)

    return processed_contour_img, processed_contours, arrow_contour

def orient_image(img, processed_contours, arrow_contour):
    # compute arrow_centroid
    M = cv.moments(arrow_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    arrow_centroid = cx, cy

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

    def rotate_image_and_contours(img, contours, angle):
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

        # rotate contours
        rotated_contours = []
        for contour in contours:
            reshaped_contour = contour.reshape(-1, 2) # reshape to 2D for easier rotation
            rotated_contour = np.matmul(M, np.vstack((reshaped_contour.T, np.ones((1, reshaped_contour.shape[0]))))).T.astype(int)
            rotated_contours.append(rotated_contour.reshape(-1, 1, 2)) # reshape back to original contour shape

        return rotated_img, rotated_contours

    closest_side = get_closest_side(img, arrow_centroid)
    angle = rotation_based_on_side(closest_side)
    rotated_img, rotated_contours = rotate_image_and_contours(img, processed_contours, angle)

    #cv.imshow("Display window", rotated_img)
    #k = cv.waitKey(0)

    return rotated_img, rotated_contours

def bisection_processing(rotated_img, rotated_contours):
    # detect extreme points
    all_extreme_points = []
    for contour in rotated_contours:
        contour = contour.reshape(-1, 2) # reshape to 2D for easier sorting
        sorted_by_x = sorted(contour, key=lambda point: point[0])
        sorted_by_y = sorted(contour, key=lambda point: point[1])

        top_25 = sorted_by_y[:25]
        right_25 = sorted_by_x[-25:]
        bottom_25 = sorted_by_y[-25:]
        left_25 = sorted_by_x[:25]
        extreme_points = [top_25, right_25, bottom_25, left_25]

        all_extreme_points.append(extreme_points)

    extreme_points_img = rotated_img.copy()
    extreme_point_thickness = 8
    for extreme_points in all_extreme_points:
        for side in extreme_points:
            for extreme_point in side:
                cv.circle(extreme_points_img, extreme_point, extreme_point_thickness, (0, 0, 255), -1)

    #cv.imshow("Display window", extreme_points_img)
    #k = cv.waitKey(0)

    # split up each set of extreme points into three groups (left, bisection, right) and find representative point
    final_groups = []
    for extreme_points in all_extreme_points:
        flattened_extreme_points = [extreme_point for side in extreme_points for extreme_point in side]
        
        clustering = DBSCAN(eps=300, min_samples=2).fit(flattened_extreme_points) 
        labels = clustering.labels_

        # separate points into groups
        grouped_points = {i: [] for i in range(max(labels)+1)}
        for label, point in zip(labels, flattened_extreme_points):
            if label != -1: # ignore noise points
                grouped_points[label].append(point)

        sorted_groups = sorted(list(grouped_points.values()), key=lambda group: np.mean([point[0] for point in group])) # sort groups by their mean x coordinate 
        mean_points = {position: np.mean(group, axis=0) for position, group in zip(["left", "bisection", "right"], sorted_groups)} # calculate mean point for each group
        
        # set bisection y-value more accurately
        mean_y = int((mean_points["left"][1] + mean_points["right"][1]) / 2)
        mean_points["bisection"][1] = mean_y

        final_groups.append(mean_points)

    final_groups = sorted(final_groups, key=lambda group: group["bisection"][1]) # sort final groups by the y coordinate of their bisection

    bisection_img = rotated_img.copy()
    line_thickness = 3
    for cross in final_groups:
        for position, size, colour in zip(["left", "bisection", "right"], [30, 45, 30], [(0, 0, 0), (0, 0, 255), (0, 0, 0)]):
            x, y = cross[position].astype(int)
            cv.line(bisection_img, (x, y - size), (x, y + size), colour, line_thickness)
    
    #cv.imshow("Display window", bisection_img)
    #k = cv.waitKey(0)

    return bisection_img, final_groups

def post_processing(bisection_img, final_groups):
    def score_calculation(bisection_x, ranges, mean_x):
        if ranges["three_point"][0][0] <= bisection_x <= ranges["three_point"][1][0]:
            if bisection_x <= mean_x:
                return 3, "L"
            else:
                return 3, "R"
        elif ranges["two_point"][0][0] <= bisection_x <= ranges["two_point"][1][0]:
            if bisection_x <= mean_x:
                return 2, "L"
            else:
                return 2, "R"
        elif ranges["one_point"][0][0] <= bisection_x <= ranges["one_point"][1][0]:
            if bisection_x <= mean_x:
                return 1, "L"
            else:
                return 1, "R"
        else:
            if bisection_x <= mean_x:
                return 0, "L"
            else:
                return 0, "R"
    
    # calculate and draw score ranges on crosses
    scoring_img = bisection_img.copy()
    LineB_T, LineB_M, LineB_B = 0, 0, 0
    for i, cross in enumerate(final_groups):
        line_length = cross["right"][0] - cross["left"][0]
        x_translation = cross["left"][0]
        mean_x = int(line_length / 2 + x_translation)
        mean_y = int((cross["left"][1] + cross["right"][1]) / 2)
        ranges = {"one_point": [(int(0.37 * line_length + x_translation), mean_y), (int(0.63 * line_length + x_translation), mean_y)],
                  "two_point": [(int(0.40 * line_length + x_translation), mean_y), (int(0.60 * line_length + x_translation), mean_y)],
                  "three_point": [(int(0.43 * line_length + x_translation), mean_y), (int(0.57 * line_length + x_translation), mean_y)]}

        cross_score = score_calculation(cross["bisection"][0], ranges, mean_x)
        if i == 0:
            LineB_T = cross_score
        elif i == 1:
            LineB_M = cross_score
        else:
            LineB_B = cross_score

        line_thickness = 3
        cv.line(scoring_img, (mean_x, mean_y - 30), (mean_x, mean_y + 30), (0, 0, 0), line_thickness)
        for range in ["one_point", "two_point", "three_point"]:
            for x, y in ranges[range]:
                cv.line(scoring_img, (x, y - 15), (x, y + 15), (0, 0, 0), line_thickness)
    
    #cv.imshow("Display window", scoring_img)
    #k = cv.waitKey(0)

    LineB = LineB_T[0] + LineB_M[0] + LineB_B[0]
    
    # return LineB_T, LineB_M, and LineB_B as a string with score followed by L or R side
    LineB_T = str(LineB_T[0]) + LineB_T[1]
    LineB_M = str(LineB_M[0]) + LineB_M[1]
    LineB_B = str(LineB_B[0]) + LineB_B[1]

    # determine standard value
    mapping = {0: 0.0,
               1: 1.0,
               2: 2.0,
               3: 3.0,
               4: 4.0,
               5: 5.5,
               6: 6.5,
               7: 7.5,
               8: 8.5,
               9: 10.0}

    LineB_SV = None
    for score, standard_value in mapping.items():
        if LineB == score:
            LineB_SV = standard_value
            break

    # determine horizontal centre of cancellation
    LineB_HCoC = None
    
    def normalize_bisection(bisection_x, line_start_x, line_end_x):
        # calculate the mean x of the line
        mean_x = (line_start_x + line_end_x) / 2

        # normalize bisection_x on the scale from -1 to 1
        normalized_bisection_x = 2 * ((bisection_x - line_start_x) / (line_end_x - line_start_x)) - 1

        return normalized_bisection_x

    LineB_HCoC_T = normalize_bisection(final_groups[0]['bisection'][0], final_groups[0]['left'][0], final_groups[0]['right'][0])
    LineB_HCoC_M = normalize_bisection(final_groups[1]['bisection'][0], final_groups[1]['left'][0], final_groups[1]['right'][0])
    LineB_HCoC_B = normalize_bisection(final_groups[2]['bisection'][0], final_groups[2]['left'][0], final_groups[2]['right'][0])

    LineB_HCoC = round((LineB_HCoC_T + LineB_HCoC_M + LineB_HCoC_B) / 3, 2)

    return LineB_T, LineB_M, LineB_B, LineB, LineB_SV, LineB_HCoC

def process_image(file_path):
    img = image_acquisition(file_path)
    pre_processed_img = image_pre_processing(img)
    contour_img, contours = contour_detection(img, pre_processed_img)
    processed_contour_img, processed_contours, arrow_contour = contour_processing(img, contours)
    rotated_img, rotated_contours = orient_image(img, processed_contours, arrow_contour)
    bisection_img, final_groups = bisection_processing(rotated_img, rotated_contours)
    LineB_T, LineB_M, LineB_B, LineB, LineB_SV, LineB_HCoC = post_processing(bisection_img, final_groups)
    return LineB_T, LineB_M, LineB_B, LineB, LineB_SV, LineB_HCoC

#file_path = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/patients/Braun/LineB.png"
#print(process_image(file_path))