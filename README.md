# BIT-Screening-Automation

## 1. Line Cancellation Task

### Task description:
Patients are required to detect and cross out all target lines on a page. When administering the test, the examiner demonstrates the nature of the task to the patient by crossing out two of the four lines located in the central column. The patient is then instructed to cross out all the lines they can see on the page. After the patient completes the task, the number of crossed-out lines not in the central column are counted. The maximum score is 36 (18 left, 18 right).

### diagnostics/line_cancellation_template.py description:
A scan of the template for the line cancellation task (templates/LineC_T.png) is read and preprocessed to reduce noise and emphasize target lines. The centroid coordinates of relevant detected contours are stored and carefully merged to remove instances of target lines being represented by more than one centroid. The arrow centroid is also isolated and subsequently used to orient the image. The centroids of the four lines in the central column are removed from the final list of centroid coordinates. The constant variable, LineC_T_C1, stores this final list of centroid coordinates (i.e., the centres of each target line) and is passed on to line_cancellation.py.

### diagnostics/line_cancellation.py description:
A scan of the patient's completed line cancellation task is read and heavily preprocessed to emphasize line intersection points. Once again, the centroid coordinates of relevant detected contours are stored and carefully merged to remove instances of line intersection points being represented by more than one centroid. The arrow centroid is again isolated and used to orient the image. The centroid coordinates are then cross-referenced against LineC_T_C1 to determine which target lines were detected. The target lines are post-processed to determine the number of crossed-out lines on the left and right sides (LineC_LS, LineC_RS), the total number of crossed-out lines and its corresponding standard value (LineC, LineC_SV), and the resulting horizontal and vertical centres of cancellation (LineC_HCoC, LineC_VCoC).

## 2. Letter Cancellation Task

### Task description:
Patients are required to detect and cross out all Es and Rs within a rectangular block of text on a page. After the patient completes the task, the number of crossed-out Es and Rs are counted. The maxmimum score is 40 (20 left, 20 right).

### diagnostics/letter_cancellation.py description:
A scan of the patient's completed letter cancellation task is read and preprocessed to reduce noise. The arrow contour is isolated among all detected contours and its centroid is used to orient the image. The image is then cropped to narrow in on the rectangular block of text and further denoised. The text block is precisely isolated by detecting its four corners and using them to apply a perspective transform onto a standardized image size. Treating the letters within the text block as a grid, the positions of the Es and Rs were determined manually which are used to isolate each letter within its own small image. These images are processed to remove as many irrelevant contours as possible and to centre the letter in a new image of standardized size. The contours, contour holes, and diagonal lines of each image are then used to determine which target letters were detected. The target letters are post-processed to determine the number of crossed-out letters on the left and right sides (LetC_LS, LetC_RS), the total number of crossed-out letters and its corresponding standard value (LetC, LetC_SV), and the resulting horizontal and vertical centres of cancellation (LetC_HCoC, LetC_VCoC).

## 3. Star Cancellation Task

### Task description:
...
