# BIT-Screening-Automation

## 1. Line Cancellation Task

### Overview:
Patients are required to detect and cross out all target lines on a page. When administering the test, the examiner demonstrates the nature of the task to the patient by crossing out two of the four lines located in the central column. The patient is then instructed to cross out all the lines they can see on the page. After the patient completes the task, the number of crossed-out lines not in the central column are counted. The maximum score is 36 (18 left, 18 right). Refer to templates/LineC_T.png or templates/LineC_T.pdf for a template of the task.

### diagnostics/line_cancellation_template.py description:
A scan of the template for the line cancellation task (templates/LineC_T.png) is read and preprocessed to reduce noise and emphasize target lines. The centroid coordinates of relevant detected contours are stored and carefully merged to remove instances of target lines being represented by more than one centroid. The arrow centroid is also isolated and subsequently used to orient the image. The centroids of the four lines in the central column are removed from the final list of centroid coordinates. The constant variable, LineC_T_C1, stores this final list of centroid coordinates (i.e., the centres of each target line) and is passed on to line_cancellation.py.

### diagnostics/line_cancellation.py description:
A scan of the patient's completed line cancellation task is read and heavily preprocessed to emphasize line intersection points. Once again, the centroid coordinates of relevant detected contours are stored and carefully merged to remove instances of line intersection points being represented by more than one centroid. The arrow centroid is again isolated and used to orient the image. The centroid coordinates are then cross-referenced against LineC_T_C1 to determine which target lines were detected. The target lines are post-processed to determine the number of crossed-out lines on the left and right sides (LineC_LS, LineC_RS), the total number of crossed-out lines and its corresponding standard value (LineC, LineC_SV), and the resulting horizontal and vertical centres of cancellation (LineC_HCoC, LineC_VCoC).

## 2. Letter Cancellation Task

### Overview:
Patients are required to detect and cross out all Es and Rs within a rectangular block of text on a page. After the patient completes the task, the number of crossed-out Es and Rs are counted. The maxmimum score is 40 (20 left, 20 right). Refer to templates/LetC_T.png or templates/LetC_T.pdf for a template of the task.

### diagnostics/letter_cancellation.py description:
A scan of the patient's completed letter cancellation task is read and preprocessed to reduce noise. The arrow contour is isolated among all detected contours and its centroid is used to orient the image. The image is then cropped to narrow in on the rectangular block of text and further denoised. The text block is precisely isolated by detecting its four corners and using them to apply a perspective transform onto a standardized image size. Treating the letters within the text block as a grid, the positions of the Es and Rs were determined manually which are used to isolate each letter within its own small image. These images are processed to remove as many irrelevant contours as possible and to centre the letter in a new image of standardized size. The contours, contour holes, and diagonal lines of each image are then used to determine which target letters were detected by the patient. The target letters are post-processed to determine the number of crossed-out letters on the left and right sides (LetC_LS, LetC_RS), the total number of crossed-out letters and its corresponding standard value (LetC, LetC_SV), and the resulting horizontal and vertical centres of cancellation (LetC_HCoC, LetC_VCoC).

## 3. Star Cancellation Task

### Overview:
Patients are required to detect and cross out all small stars on a page strewn with small stars, big stars, and letters. After the patient completes the task, the number of crossed-out small stars are counted, not including the two small stars in the central column. The maximum score is 54 (27 left, 27 right). Refer to templates/StarC_T.png or templates/StarC_T.pdf for a template of the task.

### diagnostics/star_cancellation.py description:
A scan of the patient's completed star cancellation task is read and preprocessed to reduce noise and erode the white background. The arrow contour is isolated among all detected contours and its centroid is used to orient the image. The image is then cropped based on the centroid of the rotated arrow contour to narrow in on the ROI. The ROI is precisely isolated by detecting four points that can be consistently identified across all patient scans and using them to apply a perspective transform onto a standardized image size. The coordinates of the target points used to define the perspective transform matrix were manually determined from templates/StarC_T_cropped.png (i.e., the ROI of templates/StarC_T.png). The coordinates of the 54 small stars were also manually determined from templates/StarC_T_cropped.png and used to isolate each small star within its own small image. These images are processed to remove as many irrelevant contours as possible and to rotate and centre the small star to standardize its orientation and position. The contours, skeleton, corners, slant, and other morphological features of each image are then used to determine which small stars were detected by the patient. The target stars are post-processed to determine the number of crossed-out small stars on the left and right sides (StarC_LS, StarC_RS), the total number of crossed-out small stars and its corresponding standard value (StarC, StarC_SV), and the resulting horizontal and vertical centres of cancellation (StarC_HCoC, StarC_VCoC).

## 4. Drawing Tasks

### Overview:
...

### diagnostics/star_drawing.py description:
...

### diagnostics/diamond_drawing.py description:
...

### diagnostics/flower_drawing.py description:
...

## 5. Line Biseection Task

### Overview
Patients are required to bisect three horizontal lines on a page. After the patient completes the task, the deviation of the patient's subjective centre from the actual centre of each line is objectively evaluated. The maximum score is 9 (3 per line). Refer to templates/LineB_T.png or templates/LineB_T.pdf for a template of the task.

### diagnostics/line_bisection.py description:
A scan of the patient's completed line bisection task is read and heavily preprocessed to reduce noise and emphasize line bisections. The centroid coordinates of relevant detected contours are stored and used to merge nearby contours. The merged contours are further filtered to retain only line bisection contours and then processed to reduce shape complexity. The arrow contour is also isolated among the merged contours and its centroid is used to orient the image. The 25 topmost, rightmost, bottommost, and leftmost points of each line bisection contour are used to define representative points for the left edge, bisection, and right edge for each of the three line bisections. These points are post-processed to visualize and score from 0–3 the deviation of the patient's subjective centre from the actual centre of the top, middle, and bottom lines (LineB_T, LineB_M, LineB_B). An 'L' or 'R' is also appended to each of these scores to indicate whether the deviation was to the left or right. The overall score out of 9 and its corresponding standard value are then determined (LineB, LineB_SV), along with the resulting horizontal centre of cancellation (LineB_HCoC).
