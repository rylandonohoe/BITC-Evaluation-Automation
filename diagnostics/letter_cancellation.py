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

    return img

def image_pre_processing(img):
    # grayscale conversion
    gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # image is now 1-channel

    #cv.imshow("gray1", gray1)
    #k = cv.waitKey(0)
    #cv.destroyWindow("gray1")

    # noise reduction
    blur = cv.fastNlMeansDenoising(gray1, None, h=25, templateWindowSize=25, searchWindowSize=25)
    
    #cv.imshow("blur", blur)
    #k = cv.waitKey(0)
    #cv.destroyWindow("blur")

    # thresholding
    ret, thresh = cv.threshold(blur, 225, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh", thresh)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh")

    pre_processed_img = thresh
    
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
                    arrow_centroid = cx, cy
    
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

    def rotate_image_and_centroids(img, angle):
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

        return rotated_img

    closest_side = get_closest_side(img, arrow_centroid)
    angle = rotation_based_on_side(closest_side)
    rotated_img = rotate_image_and_centroids(img, angle)

    #cv.imshow("rotated_img", rotated_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("rotated_img")

    return rotated_img

def image_reprocessing(rotated_img):
    height, width = rotated_img.shape[:2]
    resized = rotated_img[int(height/4 - 50):int(height/3*2 + 50), 75:int(width - 75)]
    
    #cv.imshow("resized", resized)
    #k = cv.waitKey(0)
    #cv.destroyWindow("resized")

    # grayscale conversion
    gray1 = cv.cvtColor(resized, cv.COLOR_BGR2GRAY) # image is now 1-channel

    #cv.imshow("gray1", gray1)
    #k = cv.waitKey(0)
    #cv.destroyWindow("gray1")

    # noise reduction
    blur = cv.fastNlMeansDenoising(gray1, None, h=25, templateWindowSize=25, searchWindowSize=25)
    
    #cv.imshow("blur", blur)
    #k = cv.waitKey(0)
    #cv.destroyWindow("blur")

    # thresholding
    ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

    #cv.imshow("thresh", thresh)
    #k = cv.waitKey(0)
    #cv.destroyWindow("thresh")

    reprocessed_img = thresh

    return reprocessed_img

def isolate_text_block(reprocessed_img):
    # find corners of text block
    def find_corners(img):
        # find all black pixel coordinates
        black_pixels = np.argwhere(img == 0)

        # create meshgrid of coordinates
        x = np.arange(img.shape[1])
        y = np.arange(img.shape[0])
        coords = np.meshgrid(x, y)

        # create meshgrids of distances to each corner
        top_left_distances = np.sqrt((coords[1] - 0)**2 + (coords[0] - 0)**2)
        top_right_distances = np.sqrt((coords[1] - 0)**2 + (coords[0] - (img.shape[1] - 1))**2)
        bottom_left_distances = np.sqrt((coords[1] - (img.shape[0] - 1))**2 + (coords[0] - 0)**2)
        bottom_right_distances = np.sqrt((coords[1] - (img.shape[0] - 1))**2 + (coords[0] - (img.shape[1] - 1))**2)

        # find black pixel closest to each corner
        top_left = black_pixels[np.argmin(top_left_distances[black_pixels[:,0], black_pixels[:,1]])]
        top_right = black_pixels[np.argmin(top_right_distances[black_pixels[:,0], black_pixels[:,1]])]
        bottom_left = black_pixels[np.argmin(bottom_left_distances[black_pixels[:,0], black_pixels[:,1]])]
        bottom_right = black_pixels[np.argmin(bottom_right_distances[black_pixels[:,0], black_pixels[:,1]])]

        corners = [(point[1], point[0]) for point in [top_left, top_right, bottom_left, bottom_right]]
        
        return corners
    
    # shift corners outward
    def shift_corners(corners):
        top_left, top_right, bottom_left, bottom_right = corners

        shift = 40
        top_left = (top_left[0] - shift/4, top_left[1] - shift)
        top_right = (top_right[0] + shift/4, top_right[1] - shift)
        bottom_left = (bottom_left[0] - shift/4, bottom_left[1] + shift)
        bottom_right = (bottom_right[0] + shift/4, bottom_right[1] + shift)

        shifted_corners = [(point[0], point[1]) for point in [top_left, top_right, bottom_left, bottom_right]]

        return shifted_corners

    corners = find_corners(reprocessed_img)
    shifted_corners = shift_corners(corners)

    gray3 = cv.cvtColor(reprocessed_img, cv.COLOR_GRAY2BGR) # image is now 3-channel
    corner_img = gray3.copy()
    for corner in shifted_corners:
        cv.circle(corner_img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
    
    #cv.imshow("corner_img", corner_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("corner_img")

    # apply projection based on corner coordinates
    def apply_perspective_transform(img, corners):
        # create arrays for src and dst
        top_left, top_right, bottom_left, bottom_right = corners
        src = np.array([top_left, top_right, bottom_left, bottom_right], dtype='float32')
        
        width, height = 2000, 600 # set normalized image size
        dst = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype='float32')

        # compute and apply the perspective transform matrix based on src and dst arrays
        M = cv.getPerspectiveTransform(src, dst)
        warped_img = cv.warpPerspective(img, M, (width, height))

        return warped_img

    text_img = apply_perspective_transform(gray3, shifted_corners)

    #cv.imshow("text_img", text_img)
    #k = cv.waitKey(0)
    #cv.destroyWindow("text_img")

    return text_img

def isolate_text_lines(text_img):
    height, width = text_img.shape[:2]
    text_lines = []

    for i in range(5):
        text_line = text_img[int(height/5*i):int(height/5*(i+1)), 0:int(width)]
        text_lines.append(text_line)

    #for i, line in enumerate(text_lines):
        #cv.imshow(f"Line{i}", line)
        #cv.waitKey(0)
        #cv.destroyWindow(f"Line{i}")
    
    return text_lines

def isolate_Es_and_Rs(text_lines, patient):
    # isolate letters based on letter positions in text lines
    def isolate_letters(letter_positions, letter):
        isolated_letters = []

        for line_num, text_line in enumerate(text_lines):
            height, width = text_line.shape[:2]
            for letter_num, letter_pos in enumerate(letter_positions[line_num]):
                letter_id = f"{patient}_Line{line_num}_{letter}{letter_num}"
                letter_img = text_line[0:height, int(width/34*letter_pos - 20):int(width/34*(letter_pos+1) + 20)]
                
                isolated_letter = {"id": letter_id,
                                   "type": letter,
                                   "position": (letter_pos, line_num),
                                   "crossed": False,
                                   "image": letter_img}

                isolated_letters.append(isolated_letter)

        return isolated_letters

    E_positions = {0: [1, 10, 23, 25],
                   1: [3, 11, 20, 23, 27],
                   2: [8, 21, 33],
                   3: [9, 15],
                   4: [2, 7, 14, 20, 27, 32]}

    R_positions = {0: [5, 15, 20, 29],
                   1: [8, 14, 18, 32],
                   2: [3, 19, 28],
                   3: [6, 12, 22, 30],
                   4: [5, 10, 15, 24, 30]}

    Es_and_Rs = isolate_letters(E_positions, 'E') + isolate_letters(R_positions, 'R')

    return Es_and_Rs

def letter_processing(Es_and_Rs):
    processed_Es_and_Rs = []
    normalized_size = (120, 100) # set standard letter_img size

    for letter_dict in Es_and_Rs:
        letter_id = letter_dict["id"]
        letter = letter_dict["type"]
        position = letter_dict["position"]
        crossed = letter_dict["crossed"]
        letter_img = letter_dict["image"]
        
        # grayscale conversion
        gray1 = cv.cvtColor(letter_img, cv.COLOR_BGR2GRAY) # image is now 1-channel

        #cv.imshow("gray1", gray1)
        #k = cv.waitKey(0)
        #cv.destroyWindow("gray1")

        # thresholding
        ret, thresh = cv.threshold(gray1, 250, 255, cv.THRESH_BINARY_INV)

        #cv.imshow("thresh", thresh)
        #k = cv.waitKey(0)
        #cv.destroyWindow("thresh")

        # find contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        contour_img = letter_img.copy()
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
        
        filtered_img = cv.bitwise_and(letter_img, letter_img, mask=mask) # apply the mask to letter_img
        filtered_img[mask==0] = [255, 255, 255] # set other contours to white

        # grayscale conversion
        gray1 = cv.cvtColor(filtered_img, cv.COLOR_BGR2GRAY) # image is now 1-channel

        #cv.imshow("gray1", gray1)
        #k = cv.waitKey(0)
        #cv.destroyWindow("gray1")

        # thresholding
        ret, thresh = cv.threshold(gray1, 200, 255, cv.THRESH_BINARY_INV)

        #cv.imshow("thresh", thresh)
        #k = cv.waitKey(0)
        #cv.destroyWindow("thresh")

        # find contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        contour_img = np.zeros((normalized_size[0], normalized_size[1]), dtype=np.uint8) 
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

        # centre the largest contour by defining and applying the translation matrix
        tx = normalized_size[1] // 2 - cx
        ty = normalized_size[0] // 2 - cy
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        centred_img = cv.warpAffine(contour_img, M, (normalized_size[1], normalized_size[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
        
        inverted = cv.bitwise_not(centred_img)

        #cv.imshow(letter_id + "_processed", inverted)
        #k = cv.waitKey(0)
        #cv.destroyWindow(letter_id + "_processed")

        # create a new dictionary for the processed letter dict and append to list
        processed_letter_dict = {"id": letter_id + "_processed",
                                "type": letter,
                                "position": position,
                                "crossed": crossed,
                                "image": inverted}

        processed_Es_and_Rs.append(processed_letter_dict)

    return processed_Es_and_Rs

def letter_cancellation_detection(processed_Es_and_Rs):
    final_Es_and_Rs = []

    for letter_dict in processed_Es_and_Rs:
        letter_id = letter_dict["id"]
        letter = letter_dict["type"]
        position = letter_dict["position"]
        crossed = letter_dict["crossed"]
        letter_img = letter_dict["image"]
        
        height, width = letter_img.shape[:2]
        
        # first check: contour proximity to border
        flag = False

        # noise reduction
        blur = cv.medianBlur(letter_img, 3)

        #cv.imshow("blur", blur)
        #k = cv.waitKey(0)
        #cv.destroyWindow("blur")
        
        # erosion
        element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        eroded = cv.erode(blur, element, iterations=1)
        
        #cv.imshow("eroded", eroded)
        #k = cv.waitKey(0)
        #cv.destroyWindow("eroded")
        
        # dilation
        dilated = cv.dilate(eroded, element, iterations=1)
        
        #cv.imshow("dilated", dilated)
        #k = cv.waitKey(0)
        #cv.destroyWindow("dilated")
            
        # edge detection
        lower_threshold = 100 # lower threshold value in Hysteresis Thresholding
        upper_threshold = 200 # upper threshold value in Hysteresis Thresholding
        aperture_size = 3 # aperture size of the Sobel filter
        edges = cv.Canny(dilated, lower_threshold, upper_threshold, aperture_size)
        
        #cv.imshow("edges", edges)
        #k = cv.waitKey(0)
        #cv.destroyWindow("edges")
        
        # find contours
        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        contour_img = 255 * np.ones_like(letter_img)
        cv.drawContours(contour_img, contours, -1, (0, 0, 255), 1)

        #cv.imshow("contour_img", contour_img)
        #k = cv.waitKey(0)
        #cv.destroyWindow("contour_img")
        
        centroids = []
        for j, contour in enumerate(contours):
            # check if contour is near the border which definitively means the letter was crossed
            x, y, w, h = cv.boundingRect(contour)
            if ((letter == 'E' and (0 <= y <= 28 or height - 28 <= y + h <= height)) or (letter == 'R' and (0 <= y <= 30 or height - 25 <= y + h <= height))):
                flag = True
            
            # second check: presence of child contours (holes)
            elif hierarchy[0][j][2] != -1:
                child_contour = contours[hierarchy[0][j][2]]
                area = cv.contourArea(child_contour)

                # isolate valid holes
                if (letter == 'E' and 25 <= area <= 500) or (letter == 'R' and 225 <= area <= 350):
                    M = cv.moments(child_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # check if centroid is close to any existing centroids
                        for (other_cx, other_cy) in centroids:
                            if abs(cx - other_cx) < 5 and abs(cy - other_cy) < 5:
                                # centroids are close, probably the same hole
                                break
                        else: # centroids are not close (new hole)
                            centroids.append((cx, cy))
                            cv.drawContours(contour_img, [child_contour], -1, (255, 0, 0), 1)

        # third check: presence of diagonal line
        if not flag:
            if letter == 'E':
                rois = [edges[int(height/4):int(3*height/4), int(width/4):int(3*width/4)]]
                minLineLength = 10
                line_offset = [(int(width/4), int(height/4))]
            elif letter == 'R':
                roi1 = edges[int(height/2):int(3*height/4), int(width/4):int(2*width/3)]
                roi2 = edges[int(height/4):int(height/2), int(width/3):int(2*width/3)]
                rois = [roi1, roi2]
                minLineLength = 9
                line_offset = [(int(width/4), int(height/2)), (int(width/3), int(height/4))]
                
            for k, roi in enumerate(rois):
                # find lines
                lines = cv.HoughLinesP(roi, 1, np.pi/180, threshold=10, minLineLength=minLineLength, maxLineGap=3)

                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if x2 - x1 != 0:
                            slope = (y2 - y1) / (x2 - x1)
                            if ((letter == 'E' and -2.0 <= slope <= -0.5) or (letter == 'R' and -4.0 <= slope <= -0.5)): # check if slope is within range
                                flag = True
                                x_offset, y_offset = line_offset[k]
                                cv.line(contour_img, (x1 + x_offset, y1 + y_offset), (x2 + x_offset, y2 + y_offset), (0, 255, 0), 2)
                                break
        
        # create a new dictionary for the final letter dict and append to list
        final_letter_dict = {"id": letter_id,
                             "type": letter,
                             "position": position,
                             "crossed": (flag or (letter == "E" and len(centroids) > 1) or (letter == "R" and len(centroids) != 1)),
                             "image": letter_img}

        final_Es_and_Rs.append(final_letter_dict)

    return final_Es_and_Rs

def post_processing(final_Es_and_Rs, text_img):
    grid_height, grid_width = 5, 34
    height, width = text_img.shape[:2]
    cell_height = height / grid_height
    cell_width = width / grid_width
    scoring_img = text_img.copy()
    
    LetC_LS = 0
    LetC_RS = 0
    LetC = 0
    letter_positions = []
    detected_letter_positions = []

    for letter_dict in final_Es_and_Rs:
        letter_positions.append(letter_dict["position"])

        x, y = letter_dict["position"][0], letter_dict["position"][1]
        centre_x, centre_y = int((x + 0.5) * cell_width), int((y + 0.5) * cell_height)

        if letter_dict["crossed"]:
            detected_letter_positions.append(letter_dict["position"])
            cv.circle(scoring_img, (centre_x, centre_y), 8, (0, 255, 0), -1)
            
            LetC += 1 # increase the count of crossed letters
            if 0 <= x <= 16:
                LetC_LS += 1 # increase the count of crossed letters on left side
            elif 17 <= x <= 33:
                LetC_RS += 1 # increase the count of crossed letters on right side
        else:
            cv.circle(scoring_img, (centre_x, centre_y), 8, (0, 0, 255), -1)

    #cv.imshow("scoring_img", scoring_img)
    #cv.waitKey(0)
    #cv.destroyWindow("scoring_img")
    
    # determine standard value
    mapping = {(0, 1): 0.0,
               (2, 3): 0.5,
               (4, 5): 1.0,
               (6, 7): 1.5,
               (8, 9): 2.0,
               (10, 11): 2.5,
               (12, 13): 3.0,
               (14, 15): 3.5,
               (16, 17): 4.0,
               (18, 19): 4.5,
               (20, 21): 5.0,
               (22, 23): 5.5,
               (24, 25): 6.0,
               (26, 27): 6.5,
               (28, 29): 7.0,
               (30, 31): 7.5,
               (32, 33): 8.0,
               (34, 35): 8.5,
               (36, 37): 9.0,
               (38, 39): 9.5,
               (40,): 10.0}

    LetC_SV = None
    for interval, standard_value in mapping.items():
        if len(interval) == 2:
            if interval[0] <= LetC <= interval[1]:
                LetC_SV = standard_value
                break
        elif len(interval) == 1:
            if interval[0] == LetC:
                LetC_SV = standard_value
                break
    
    # determine horizontal and vertical centres of cancellation
    def calculate_CoC(letter_positions, detected_letter_positions):
        # calculate the mean positions
        mean_x_targets, mean_y_targets = np.mean(letter_positions, axis=0)
        mean_x_detected, mean_y_detected = np.mean(detected_letter_positions, axis=0)

        # calculate the leftmost, bottommost, rightmost, and topmost targets
        leftmost_target, bottommost_target = np.min(letter_positions, axis=0)
        rightmost_target, topmost_target = np.max(letter_positions, axis=0)

        # adjust the scale so that range of targets is from -1 to 1 with the mean of targets being 0
        LetC_HCoC = round(2 * (mean_x_detected - mean_x_targets) / (rightmost_target - leftmost_target), 2)
        LetC_VCoC = round(2 * (mean_y_detected - mean_y_targets) / (topmost_target - bottommost_target), 2)
        
        return LetC_HCoC, LetC_VCoC

    LetC_HCoC, LetC_VCoC = calculate_CoC(letter_positions, detected_letter_positions)

    return LetC_LS, LetC_RS, LetC, LetC_SV, LetC_HCoC, LetC_VCoC

def process_image(file_path, patient):
    img = image_acquisition(file_path)
    pre_processed_img = image_pre_processing(img)
    contour_img, contours = contour_detection(img, pre_processed_img)
    arrow_centroid = arrow_detection(img, contours)
    rotated_img = orient_image(img, arrow_centroid)
    reprocessed_img = image_reprocessing(rotated_img)
    text_img = isolate_text_block(reprocessed_img)
    text_lines = isolate_text_lines(text_img)
    Es_and_Rs = isolate_Es_and_Rs(text_lines, patient)
    processed_Es_and_Rs = letter_processing(Es_and_Rs)
    final_Es_and_Rs = letter_cancellation_detection(processed_Es_and_Rs)
    LetC_LS, LetC_RS, LetC, LetC_SV, LetC_HCoC, LetC_VCoC = post_processing(final_Es_and_Rs, text_img)
    return LetC_LS, LetC_RS, LetC, LetC_SV, LetC_HCoC, LetC_VCoC