import cv2 as cv
import numpy as np
import skimage.morphology
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

    # noise reduction
    blur = cv.fastNlMeansDenoising(gray1, None, h=40, templateWindowSize=25, searchWindowSize=25)
    
    #cv.imshow("blur", blur)
    #k = cv.waitKey(0)
    #cv.destroyWindow("blur")

    # thresholding
    ret, thresh = cv.threshold(blur, 225, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh", thresh)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh")

    # erosion
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    eroded = cv.erode(thresh, element, iterations=1)

    #cv.imshow("eroded", eroded)
    #k = cv.waitKey(0)
    #cv.destroyWindow("eroded")

    pre_processed_img = eroded

    return pre_processed_img

def contour_detection(img, pre_processed_img):
    contours, hierarchy = cv.findContours(pre_processed_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    contour_img = img.copy()
    cv.drawContours(contour_img, contours, -1, (0, 255, 0), 5)

    #cv.imshow("contour_img", contour_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("contour_img")

    return contour_img, contours

def arrow_detection(img, contours):
    arrow_centroid = None

    height, width = img.shape[:2]
    border_buffer = 75
    max_distance = 0
    
    filtered_centroids = []
    
    # find centroids of valid contours
    for contour in contours:
        M = cv.moments(contour)
        area = cv.contourArea(contour)
        if M['m00'] != 0 and 0 < area < 10000:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            distance_to_border = min(cx, cy, width - cx, height - cy)
            if distance_to_border > border_buffer and (((width/2 - 150) < cx < (width/2 + 150)) or ((height/2 - 150) < cy < (height/2 + 150))): # avoid border contours caused by scanning
                filtered_centroids.append((cx, cy))
    
    # find the contour whose centroid is farthest away from another centroid
    for i, centroid in enumerate(filtered_centroids):
        min_distance = min(np.sqrt((centroid[0] - cx)**2 + (centroid[1] - cy)**2) for j, (cx, cy) in enumerate(filtered_centroids) if i != j)
        if min_distance > max_distance:
            max_distance = min_distance
            arrow_centroid = centroid
    
    arrow_img = img.copy()
    cv.circle(arrow_img, arrow_centroid, 8, (255, 0, 0), -1)

    #cv.imshow("arrow_img", arrow_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("arrow_img")

    return arrow_centroid

def orient_image(img, arrow_centroid):
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

    def rotate_image_and_centroids(img, angle, arrow_centroid):
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

        # rotate arrow centroid
        arrow_centroid = np.array([[arrow_centroid[0]], [arrow_centroid[1]], [1]])
        rotated_arrow_centroid = np.matmul(M, arrow_centroid)
        rotated_arrow_centroid = (int(rotated_arrow_centroid[0][0]), int(rotated_arrow_centroid[1][0]))

        return rotated_img, rotated_arrow_centroid

    closest_side = get_closest_side(img, arrow_centroid)
    angle = rotation_based_on_side(closest_side)
    rotated_img, rotated_arrow_centroid = rotate_image_and_centroids(img, angle, arrow_centroid)

    #cv.imshow("rotated_img", rotated_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("rotated_img")

    return rotated_img, rotated_arrow_centroid

def image_reprocessing(rotated_img, rotated_arrow_centroid):
    height, width = rotated_img.shape[:2]
    cx, cy = rotated_arrow_centroid

    # define ROI
    bottom = cy - (6*height/100)
    top = bottom - (4*height/5)
    left = cx - (87*width/200)
    right = cx + (52*width/100)

    resized = rotated_img[int(top):int(bottom), int(left):int(right)]
    
    #cv.imshow("resized", resized)
    #k = cv.waitKey(0)
    #cv.destroyWindow("resized")

    # grayscale conversion
    gray1 = cv.cvtColor(resized, cv.COLOR_BGR2GRAY) # image is now 1-channel

    #cv.imshow("gray1", gray1)
    #k = cv.waitKey(0)
    #cv.destroyWindow("gray1")

    # thresholding
    ret, thresh = cv.threshold(gray1, 225, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh", thresh)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh")

    reprocessed_img = thresh

    return reprocessed_img

def normalize_img(reprocessed_img):
    # noise reduction
    blur = cv.fastNlMeansDenoising(reprocessed_img, None, h=35, templateWindowSize=25, searchWindowSize=25)
    
    #cv.imshow("blur", blur)
    #k = cv.waitKey(0)
    #cv.destroyWindow("blur")

    # thresholding
    ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh", thresh)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh")

    # find four consistent points to later apply a perspective transform
    height, width = thresh.shape[:2]
    
    second_quarter = thresh[int(height/4):int(height/2), :]
    second_quarter_black_pixels = np.argwhere(second_quarter == 0)
    point1_index = second_quarter_black_pixels[:, 1].argmin()
    point1 = second_quarter_black_pixels[point1_index][::-1] # reverse order to get (x, y)
    point1[1] += int(height/4)

    second_twelfth = thresh[:, int(width/12):int(2*width/12)]
    second_twelfth_black_pixels = np.argwhere(second_twelfth == 0)
    point2_index = second_twelfth_black_pixels[:, 0].argmax()
    point2 = second_twelfth_black_pixels[point2_index][::-1] # reverse order to get (x, y)
    point2[0] += int(width/12)

    eleventh_twenty_fourth_top = thresh[100:int(height/2), int(11*width/24):int(12*width/24)]
    eleventh_twenty_fourth_top_black_pixels = np.argwhere(eleventh_twenty_fourth_top == 0)
    point3_index = eleventh_twenty_fourth_top_black_pixels[:, 0].argmin()
    point3 = eleventh_twenty_fourth_top_black_pixels[point3_index][::-1] # reverse order to get (x, y)
    point3[0] += int(11*width/24)
    point3[1] += 100

    middle_twelfth_bottom = thresh[int(height/2):, int(5.5*width/12):int(6.5*width/12)]
    middle_bottom_black_pixels = np.argwhere(middle_twelfth_bottom == 0)
    max_y_value = np.max(middle_bottom_black_pixels[:, 0])
    point4_pixels = middle_bottom_black_pixels[middle_bottom_black_pixels[:, 0] >= max_y_value - 3]
    point4 = point4_pixels[point4_pixels[:, 1].argmin()]
    point4 = [point4[1] + int(5.5*width/12), point4[0] + int(height/2)]

    source_points = np.array([point1, point2, point3, point4], dtype='float32')
    
    extremity_img = cv.cvtColor(reprocessed_img, cv.COLOR_GRAY2RGB)
    for point in source_points:
        cv.circle(extremity_img, tuple(point.astype(int)), 8, (0, 0, 255), -1)
    
    #cv.imshow("extremity_img", extremity_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("extremity_img")

    # define target points based on StarC_T_cropped.png template
    target_points = np.array([[25, 435], [255, 1285], [1045, 145], [1090, 1170]], dtype='float32')

    # compute and apply perspective transformation matrix
    M = cv.getPerspectiveTransform(source_points, target_points)

    height, width = 1315, 2175 # based on StarC_T_cropped.png template
    normalized_img = np.zeros((height, width), dtype=np.uint8)
    normalized_img = cv.warpPerspective(reprocessed_img, M, (width, height), borderValue=255)

    #cv.imshow("normalized_img", normalized_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("normalized_img")

    return normalized_img

def isolate_stars(normalized_img, patient):
    # star coordinates based on StarC_T_cropped.png template (ignoring middle two)
    star_coordinates = {0: [(70, 140), (135, 305), (185, 365), (145, 435), (100, 510), (100, 705), (140, 925), (125, 1030)],
                        1: [(435, 220), (425, 435), (430, 640), (360, 770), (420, 820), (485, 970), (410, 1030), (475, 1240)],
                        2: [(805, 160), (745, 220), (860, 220), (730, 435), (800, 555), (710, 640), (775, 800), (725, 840), (875, 840), (790, 880), (855, 1160)],
                        3: [(1210, 175), (1165, 215), (1290, 225), (1355, 390), (1200, 520), (1355, 585), (1290, 655), (1130, 855), (1200, 930), (1070, 980), (1150, 1060)],
                        4: [(1665, 175), (1655, 385), (1585, 450), (1740, 450), (1585, 845), (1775, 1000), (1710, 1080), (1535, 1185)],
                        5: [(1995, 120), (1980, 225), (2030, 295), (2025, 435), (1985, 535), (2000, 650), (2120, 780), (2090, 935)]}
    
    stars = []
    for column, column_coordinates in star_coordinates.items():
        for index, coordinates in enumerate(column_coordinates):
            x, y = coordinates

            star_id = f"{patient}_Col{column}_{index}"
            side = 'L' if column < len(star_coordinates) / 2 else 'R'
            
            # set the borders for cropping the image
            x_min = max(0, x - 60)
            x_max = min(normalized_img.shape[1], x + 60)
            y_min = max(0, y - 60)
            y_max = min(normalized_img.shape[0], y + 60)

            star_img = normalized_img[y_min:y_max, x_min:x_max]

            star = {"id": star_id,
                    "coordinates": coordinates,
                    "side": side,
                    "crossed": False,
                    "image": star_img}
            
            stars.append(star)

    return stars

def star_processing(stars):
    processed_stars = []

    for star_dict in stars:
        star_id = star_dict["id"]
        coordinates = star_dict["coordinates"]
        side = star_dict["side"]
        crossed = star_dict["crossed"]
        star_img = star_dict["image"]

        height, width = star_img.shape[:2]

        # thresholding
        ret, thresh = cv.threshold(star_img, 250, 255, cv.THRESH_BINARY_INV)

        #cv.imshow("thresh", thresh)
        #k = cv.waitKey(0)
        #cv.destroyWindow("thresh")

        # find contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        contour_img = star_img.copy()
        cv.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

        #cv.imshow("contour_img", contour_img)
        #k = cv.waitKey(0)
        #cv.destroyWindow("contour_img")
        
        # remove unnecessary contours
        height, width = thresh.shape[:2]
        centre = [int(width/2), int(height/2)]

        mask = np.zeros_like(thresh) # create a black image with the same dimensions as thresh
        
        for contour in contours:
            M = cv.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if abs(centre[0] - cx) < 30 and abs(centre[1] - cy) < 30: # check if contour centroid is near the centre
                    cv.drawContours(mask, [contour], -1, 255, -1)
        
        filtered_img = cv.bitwise_and(star_img, star_img, mask=mask) # apply the mask to star_img
        filtered_img[mask==0] = 255 # set other contours to white

        # thresholding
        ret, thresh = cv.threshold(filtered_img, 200, 255, cv.THRESH_BINARY_INV)

        #cv.imshow("thresh", thresh)
        #k = cv.waitKey(0)
        #cv.destroyWindow("thresh")
        
        # find contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        contour_img = np.zeros((height, width), dtype=np.uint8) 
        cv.drawContours(contour_img, contours, -1, 255, -1)

        #cv.imshow("contour_img", contour_img)
        #k = cv.waitKey(0)
        #cv.destroyWindow("contour_img")

        # find the largest contour based on area
        largest_contour = max(contours, key=cv.contourArea)

        # compute the centroid of the largest contour
        M = cv.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        # find topmost white pixel
        white_pixels = np.argwhere(thresh == 255)
        topmost = white_pixels[white_pixels[:, 0].argmin()][::-1]

        # compute and correct rotation angle
        dx = topmost[0] - cx
        dy = topmost[1] - cy
        angle = np.degrees(np.arctan2(dy, dx))

        angle = (angle - 90) % 180
        if angle > 90:
            angle -= 180

        # rotate image
        M = cv.getRotationMatrix2D((cx, cy), angle, 1)
        rotated_img = cv.warpAffine(thresh, M, (width, height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)

        #cv.imshow("rotated_img", rotated_img)
        #k = cv.waitKey(0)
        #cv.destroyWindow("rotated_img")

        # centre the largest contour by defining and applying the translation matrix
        tx = width // 2 - cx
        ty = height // 2 - cy
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        centred_img = cv.warpAffine(rotated_img, M, (width, height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
        
        inverted = cv.bitwise_not(centred_img)
        
        #cv.imshow(star_id + "_processed", inverted)
        #k = cv.waitKey(0)
        #cv.destroyWindow(star_id + "_processed")

        # create a new dictionary for the processed letter dict and append to list
        processed_letter_dict = {"id": star_id + "_processed",
                                "coordinates": coordinates,
                                "side": side,
                                "crossed": crossed,
                                "image": inverted}

        processed_stars.append(processed_letter_dict)

    return processed_stars

def star_cancellation_detection(processed_stars):
    final_stars = []
    
    for star_dict in processed_stars:
        star_id = star_dict["id"]
        coordinates = star_dict["coordinates"]
        side = star_dict["side"]
        crossed = star_dict["crossed"]
        star_img = star_dict["image"]

        height, width = star_img.shape[:2]

        # first check: contour proximity to border
        flag = False
        
        # edge detection
        lower_threshold = 100 # lower threshold value in Hysteresis Thresholding
        upper_threshold = 200 # upper threshold value in Hysteresis Thresholding
        aperture_size = 3 # aperture size of the Sobel filter
        edges = cv.Canny(star_img, lower_threshold, upper_threshold, aperture_size)
        
        #cv.imshow("edges", edges)
        #k = cv.waitKey(0)
        #cv.destroyWindow("edges")
        
        # find contours
        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        contour_img = 255 * np.ones_like(star_img)
        cv.drawContours(contour_img, contours, -1, 0, 1)

        #cv.imshow("contour_img", contour_img)
        #k = cv.waitKey(0)
        #cv.destroyWindow("contour_img")
        
        for j, contour in enumerate(contours):
            # check if contour is near the border which definitively means the letter was crossed
            x, y, w, h = cv.boundingRect(contour)
            if 0 <= x <= 22 or width - 20 <= x + w <= width or 0 <= y <= 19 or height - 23 <= y + h <= height:
                flag = True

        # second check: number of endpoints of skeleton
        if not flag:
            # erosion
            element = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
            eroded = cv.erode(star_img, element, iterations=1)

            #cv.imshow("eroded", eroded)
            #k = cv.waitKey(0)
            #cv.destroyWindow("eroded")

            # binarization and skeletonization
            binary = eroded == 0
            skeleton = skimage.morphology.skeletonize(binary)
            skeleton_img = cv.cvtColor(skeleton.astype(np.uint8)*255, cv.COLOR_GRAY2BGR)

            #cv.imshow("skeleton_img", skeleton_img)
            #k = cv.waitKey(0)
            #cv.destroyWindow("skeleton_img")

            # determine endpoints
            kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
            endpoint_coordinates = np.array([coordinates[::-1] for coordinates in np.argwhere(cv.filter2D(skeleton.astype(np.uint8), -1, kernel) == 11)]) # coordinates where value is 11 (endpoint in skeletonized image)

            # draw the endpoints
            endpoints_img = cv.cvtColor(star_img.copy(), cv.COLOR_GRAY2BGR)
            for x, y in endpoint_coordinates:
                cv.circle(endpoints_img, (x, y), 3, (0, 0, 255), -1)

            #cv.imshow("endpoints_img", endpoints_img)
            #k = cv.waitKey(0)
            #cv.destroyWindow("endpoints_img")

            centre = np.array([star_img.shape[1] // 2, star_img.shape[0] // 2])

            # determine valid endpoints (at least 25 pixels away from the centre)
            mask = np.sqrt(np.sum((endpoint_coordinates - centre)**2, axis=1)) >= 25
            valid_endpoints = endpoint_coordinates[mask]

            # merge endpoints
            merged_endpoints = []
            clustering = DBSCAN(eps=8, min_samples=1).fit(valid_endpoints)
            labels = clustering.labels_

            for label in set(labels):
                if label != -1: # ignore noise points
                    cluster_points = valid_endpoints[labels == label]
                    mean_endpoint = np.round(np.mean(cluster_points, axis=0)).astype(int)
                    merged_endpoints.append(tuple(mean_endpoint))

            merged_img = cv.cvtColor(star_img.copy(), cv.COLOR_GRAY2BGR)
            for endpoint in merged_endpoints:
                cv.circle(merged_img, (endpoint[0], endpoint[1]), 3, (0, 255, 0), -1)
            
            #cv.imshow("merged_img", merged_img)
            #k = cv.waitKey(0)
            #cv.destroyWindow("merged_img")

            if len(merged_endpoints) > 5:
                flag = True

        # third check: number of corners
        if not flag:
            # erosion
            element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            eroded = cv.erode(star_img, element, iterations=1)

            #cv.imshow("eroded", eroded)
            #k = cv.waitKey(0)
            #cv.destroyWindow("eroded")

            # corner detection
            blockSize = 10 # size of neighbourhood considered for corner detection
            kSize = 31 # aperture parameter of the Sobel derivative used
            k = 0.20 # Harris detector free parameter in the equation
            dst = cv.cornerHarris(eroded, blockSize, kSize, k)
            corners = np.where(dst > 0.02 * dst.max())
            
            corner_img = cv.cvtColor(star_img.copy(), cv.COLOR_GRAY2BGR)
            corner_img[corners] = [0, 0, 255]

            #cv.imshow("corner_img", corner_img)
            #k = cv.waitKey(0)
            #cv.destroyWindow("corner_img")

            # merge corners
            merged_corners = []
            corner_coordinates = np.array(list(zip(corners[1], corners[0]))) # format corners into list of (x, y) coordinates
            clustering = DBSCAN(eps=5, min_samples=18).fit(corner_coordinates)
            labels = clustering.labels_

            for label in set(labels):
                if label != -1: # ignore noise points
                    cluster_points = corner_coordinates[labels == label]
                    mean_corner = np.mean(cluster_points, axis=0)
                    merged_corners.append(tuple(mean_corner))
            
            merged_img = cv.cvtColor(star_img.copy(), cv.COLOR_GRAY2BGR)
            for corner in merged_corners:
                cv.circle(merged_img, (int(corner[0]), int(corner[1])), 3, (0, 255, 0), -1)
            
            #cv.imshow("merged_img", merged_img)
            #k = cv.waitKey(0)
            #cv.destroyWindow("merged_img")

            if len(merged_corners) != 5:
                flag = True
        
        # fourth check: slant detection
        if not flag:
            top_half = star_img[:height // 2, :]
            bottom_third = star_img[2 * height // 3:, :]

            for i, portion in enumerate([top_half, bottom_third]):
                black_pixels = np.where(portion == 0)
                black_pixels_coordinates = np.array(list(zip(black_pixels[1], black_pixels[0]))) # format black_pixels into list of (x, y) coordinates

                leftmost_point = black_pixels_coordinates[np.argmin(black_pixels_coordinates[:, 0])]
                rightmost_point = black_pixels_coordinates[np.argmax(black_pixels_coordinates[:, 0])]

                leftmost_x, leftmost_y = leftmost_point
                rightmost_x, rightmost_y = rightmost_point

                if i == 1:
                    leftmost_y += 2 * height // 3
                    rightmost_y += 2 * height // 3

                orientation_img = cv.cvtColor(star_img.copy(), cv.COLOR_GRAY2BGR)
                cv.circle(orientation_img, (leftmost_x, leftmost_y), 3, (0, 0, 255), -1)
                cv.circle(orientation_img, (rightmost_x, rightmost_y), 3, (0, 0, 255), -1)

                #cv.imshow("orientation_img", orientation_img)
                #k = cv.waitKey(0)
                #cv.destroyWindow("orientation_img")

                y_difference = np.abs(rightmost_y - leftmost_y)

                if (i == 0 and y_difference >= 10) or (i == 1 and y_difference >= 6):
                    flag = True
                    break
        
        # fifth check: black pixel count of image centre
        if not flag:
            # resize
            resized = star_img[3*height//8:5*height//8, 3*width//8:5*width//8]

            #cv.imshow("resized", resized)
            #k = cv.waitKey(0)
            #cv.destroyWindow("resized")

            black_pixel_num = np.sum(resized == 0)
            if black_pixel_num > 815:
                flag = True

        # sixth check: presence of top isosceles triangle
        if not flag:
            black_pixels = np.where(star_img == 0)
            black_pixels_coordinates = np.array(list(zip(black_pixels[1], black_pixels[0]))) # format black_pixels into list of (x, y) coordinates

            topmost_point = black_pixels_coordinates[np.argmin(black_pixels_coordinates[:, 1])]
            leftmost_point = black_pixels_coordinates[np.argmin(black_pixels_coordinates[:, 0])]
            rightmost_point = black_pixels_coordinates[np.argmax(black_pixels_coordinates[:, 0])]

            distance_top_left = np.linalg.norm(np.array(topmost_point) - np.array(leftmost_point))
            distance_top_right = np.linalg.norm(np.array(topmost_point) - np.array(rightmost_point))

            triangle_img = cv.cvtColor(star_img.copy(), cv.COLOR_GRAY2BGR)
            cv.line(triangle_img, tuple(leftmost_point), tuple(topmost_point), (0, 0, 255), 2)
            cv.line(triangle_img, tuple(rightmost_point), tuple(topmost_point), (0, 0, 255), 2)
            cv.line(triangle_img, tuple(leftmost_point), tuple(rightmost_point), (0, 0, 255), 2)

            #cv.imshow("triangle_img", triangle_img)
            #k = cv.waitKey(0)
            #cv.destroyWindow("triangle_img")

            distance_difference = np.abs(distance_top_left - distance_top_right)
            if distance_difference > 8:
                flag = True

        # seventh check: location of bottom middle vertex
        if not flag:
            roi_width_start = width // 2 - width // 50
            roi_width_end = width // 2 + width // 50
            roi = star_img[:, roi_width_start:roi_width_end]

            black_pixels = np.where(roi == 0)
            black_pixels_coordinates = np.array(list(zip(black_pixels[1] + roi_width_start, black_pixels[0]))) # format black_pixels into list of (x, y) coordinates and adjust x-coordinates

            bottommost_point = black_pixels_coordinates[np.argmax(black_pixels_coordinates[:, 1])]
            
            bottom_img = cv.cvtColor(star_img.copy(), cv.COLOR_GRAY2BGR)
            cv.circle(bottom_img, tuple(bottommost_point), 3, (0, 0, 255), -1)

            #cv.imshow("bottom_img", bottom_img)
            #k = cv.waitKey(0)
            #cv.destroyWindow("bottom_img")
            
            if bottommost_point[1] > 78:
                flag = True

        # create a new dictionary for the final star dict and append to list
        final_star_dict = {"id": star_id,
                            "coordinates": coordinates,
                            "side": side,
                            "crossed": flag,
                            "image": star_img}

        final_stars.append(final_star_dict)

    return final_stars

def post_processing(final_stars, normalized_img):
    height, width = normalized_img.shape[:2]
    scoring_img = cv.cvtColor(normalized_img, cv.COLOR_GRAY2BGR)
    
    StarC_LS = 0
    StarC_RS = 0
    StarC = 0
    star_coordinates = []
    detected_star_coordinates = []

    for star_dict in final_stars:
        star_coordinates.append(star_dict["coordinates"])
        x, y = star_dict["coordinates"]

        if star_dict["crossed"]:
            detected_star_coordinates.append(star_dict["coordinates"])
            cv.circle(scoring_img, (x, y), 8, (0, 255, 0), -1)
            
            StarC += 1 # increase the count of crossed stars
            if star_dict["side"] == 'L':
                StarC_LS += 1 # increase the count of crossed stars on left side
            else:
                StarC_RS += 1 # increase the count of crossed stars on right side
        else:
            cv.circle(scoring_img, (x, y), 8, (0, 0, 255), -1)

    # draw on uncounted small stars for visualization
    cv.circle(scoring_img, (1010, 645), 8, (0, 255, 255), -1)
    cv.circle(scoring_img, (1010, 1060), 8, (0, 255, 255), -1)

    #cv.imshow("scoring_img", scoring_img)
    #cv.waitKey(0)
    #cv.destroyWindow("scoring_img")
    
    # determine standard value
    mapping = {(0, 2): 0.0,
               (3, 5): 0.5,
               (6, 8): 1.0,
               (9, 10): 1.5,
               (11, 13): 2.0,
               (14, 16): 2.5,
               (17, 18): 3.0,
               (19, 21): 3.5,
               (22, 24): 4.0,
               (25, 26): 4.5,
               (27, 29): 5.0,
               (30, 32): 5.5,
               (33, 35): 6.0,
               (36, 37): 6.5,
               (38, 40): 7.0,
               (41, 43): 7.5,
               (44, 45): 8.0,
               (46, 48): 8.5,
               (49, 51): 9.0,
               (52, 53): 9.5,
               (54,): 10.0}

    StarC_SV = None
    for interval, standard_value in mapping.items():
        if len(interval) == 2:
            if interval[0] <= StarC <= interval[1]:
                StarC_SV = standard_value
                break
        elif len(interval) == 1:
            if interval[0] == StarC:
                StarC_SV = standard_value
                break
    
    # determine horizontal and vertical centres of cancellation
    def calculate_CoC(star_coordinates, detected_star_coordinates):
        # calculate the mean positions
        mean_x_targets, mean_y_targets = np.mean(star_coordinates, axis=0)
        mean_x_detected, mean_y_detected = np.mean(detected_star_coordinates, axis=0)

        # calculate the leftmost, bottommost, rightmost, and topmost targets
        leftmost_target, bottommost_target = np.min(star_coordinates, axis=0)
        rightmost_target, topmost_target = np.max(star_coordinates, axis=0)

        # adjust the scale so that range of targets is from -1 to 1 with the mean of targets being 0
        StarC_HCoC = round(2 * (mean_x_detected - mean_x_targets) / (rightmost_target - leftmost_target), 2)
        StarC_VCoC = round(2 * (mean_y_detected - mean_y_targets) / (topmost_target - bottommost_target), 2)
        
        return StarC_HCoC, StarC_VCoC

    StarC_HCoC, StarC_VCoC = calculate_CoC(star_coordinates, detected_star_coordinates)

    return StarC_LS, StarC_RS, StarC, StarC_SV, StarC_HCoC, StarC_VCoC

def process_image(file_path, patient):
    img = image_acquisition(file_path)
    pre_processed_img = image_pre_processing(img)
    contour_img, contours = contour_detection(img, pre_processed_img)
    arrow_centroid = arrow_detection(img, contours)
    rotated_img, rotated_arrow_centroid = orient_image(img, arrow_centroid)
    reprocessed_img = image_reprocessing(rotated_img, rotated_arrow_centroid)
    normalized_img = normalize_img(reprocessed_img)
    stars = isolate_stars(normalized_img, patient) # note: only small stars are being isolated
    processed_stars = star_processing(stars)
    final_stars = star_cancellation_detection(processed_stars)
    StarC_LS, StarC_RS, StarC, StarC_SV, StarC_HCoC, StarC_VCoC = post_processing(final_stars, normalized_img)
    return StarC_LS, StarC_RS, StarC, StarC_SV, StarC_HCoC, StarC_VCoC