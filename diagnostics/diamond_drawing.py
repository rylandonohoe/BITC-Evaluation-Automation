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
    # isolate diamond
    height, width = img.shape[:2]
    resized = img[int(height/3 + 75):int(height/3*2 - 75), int(width/2):int(width - 150)]

    cv.imshow("Display window", resized)
    k = cv.waitKey(0)

    # grayscale conversion
    gray1 = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    cv.imshow("Display window", gray1)
    k = cv.waitKey(0)

    # binarization (thresholding)
    ret, thresh = cv.threshold(gray1, 127, 255, cv.THRESH_BINARY)

    cv.imshow("Display window", thresh)
    k = cv.waitKey(0)

    # noise reduction (Gaussian blur)
    kernel_size = (5, 5) # larger = blurrier
    blur = cv.GaussianBlur(thresh, kernel_size, 0)

    cv.imshow("Display window", blur)
    k = cv.waitKey(0)

    # edge detection
    lower_threshold = 50 # lower threshold value in Hysteresis Thresholding
    upper_threshold = 150 # upper threshold value in Hysteresis Thresholding
    aperture_size = 3 # aperture size of the Sobel filter
    edges = cv.Canny(blur, lower_threshold, upper_threshold, aperture_size)

    cv.imshow("Display window", edges)
    k = cv.waitKey(0)

    # dilation
    kernel = np.ones((5, 5))
    dilated = cv.dilate(edges, kernel, iterations=4)

    cv.imshow("Display window", dilated)
    k = cv.waitKey(0)

    # erosion
    kernel = np.ones((5, 5))
    eroded = cv.erode(dilated, kernel, iterations=4)

    cv.imshow("Display window", eroded)
    k = cv.waitKey(0)

    return eroded

def line_detection(eroded):
    line_img = np.zeros((eroded.shape[0], eroded.shape[1], 3), dtype=np.uint8) # blank image with same size as blur_gray3
    line_img[:] = (255, 255, 255) # background of line_img set to white

    # line detection
    rho = 1 # distance resolution
    theta = np.pi / 180 # angular resolution
    threshold = 10 # minimum number of intersections to detect a line
    min_line_length = 5 # minimum number of points to form a line
    max_line_gap = 5 # maximum gap between points to be considered in the same line
    lines = cv.HoughLinesP(eroded, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    formatted_lines = [[[x1, y1], [x2, y2]] for [[x1, y1, x2, y2]] in lines]

    # draw lines on line_img
    line_thickness = 5
    for line in formatted_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

    cv.imshow("Display window", line_img)
    k = cv.waitKey(0)

    return line_img, formatted_lines

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

    # draw corners on line_img
    corner_thickness = 8
    for corner in corners:
        cv.circle(line_img, corner, corner_thickness, (0, 0, 255), -1)
    corner_img = line_img

    cv.imshow("Display window", corner_img)
    k = cv.waitKey(0)

    # construct diamond on corner_img
    line_thickness = 1
    colour = (255, 0, 0)
    cv.line(corner_img, tuple(corners[0]), tuple(corners[1]), colour, line_thickness) # top to right
    cv.line(corner_img, tuple(corners[1]), tuple(corners[2]), colour, line_thickness) # right to bottom
    cv.line(corner_img, tuple(corners[2]), tuple(corners[3]), colour, line_thickness) # bottom to left
    cv.line(corner_img, tuple(corners[3]), tuple(corners[0]), colour, line_thickness) # left to top
    cv.line(corner_img, tuple(corners[0]), tuple(corners[2]), colour, line_thickness) # top to bottom
    diamond_img = corner_img

    cv.imshow("Display window", diamond_img)
    k = cv.waitKey(0)

    return diamond_img, corners

def diamond_overlap(diamond_img, formatted_lines, corners, num_points=1000, tolerance=50):
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
    ideal_lines = [interpolate_points(corners[i], corners[(i+1)%4], num_points) for i in range(4)] + [interpolate_points(corners[0], corners[2], num_points)]

    # determine valid pixels on ideal diamond
    line_names = ["top_right", "right_bottom", "bottom_left", "left_top", "top_bottom"]
    valid_diamond_pixels = {line_names[i]: [] for i in range(5)}
    
    for i, ideal_line in enumerate(ideal_lines):
        # calculate dx and dy of ideal line
        delta_x, delta_y = ideal_line[1][0] - ideal_line[0][0], ideal_line[1][1] - ideal_line[0][1]
        dx = delta_x / np.sqrt(delta_x**2 + delta_y**2)
        dy = delta_y / np.sqrt(delta_x**2 + delta_y**2)
        
        for point in ideal_line:
            # calculate perpendicular line
            perpendicular_line = calculate_perpendicular_line(point, dx, dy, tolerance)

            # check if any of the formatted_lines intersect the perpendicular line
            for formatted_line in formatted_lines:
                intersection = line_intersection(formatted_line, perpendicular_line)
                if intersection is not None:
                    valid_diamond_pixels[line_names[i]].append(point)
                    break

    # draw valid diamond pixels on diamond_img
    pixel_thickness = 1
    for valid_line_pixels in valid_diamond_pixels.values():
        for pixel in valid_line_pixels:
            cv.circle(diamond_img, tuple(map(int, pixel)), pixel_thickness, (0, 0, 255), -1)
    overlap_img = diamond_img
    
    cv.imshow("Display window", overlap_img)
    k = cv.waitKey(0)

    # determine valid lines
    validity_threshold = 0.8
    valid_lines = {key: len(value)/num_points >= validity_threshold for key, value in valid_diamond_pixels.items()}

    return overlap_img, valid_lines

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



def process_image(file_path):
    img = image_acquisition(file_path)
    erode = image_pre_processing(img)
    line_img, formatted_lines = line_detection(erode)
    diamond_img, corners = ideal_diamond(line_img, formatted_lines)
    overlap_img, valid_lines = diamond_overlap(diamond_img, formatted_lines, corners)
    form_score = form_evaluation(valid_lines)
    detail_score = detail_evaluation(valid_lines)
    arrangement_score = arrangement_evaluation(corners)
    final_score = form_score + detail_score + arrangement_score
    return final_score

file_path = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/patients/Gerke/DrawDiamond.png"
print(process_image(file_path))