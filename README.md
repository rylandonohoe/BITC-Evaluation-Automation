# BIT-Screening-Automation

## 1. Line Cancellation Task

### Description of task:
Patients are required to detect and cross out all target lines on a page. When administering the test, the examiner demonstrates the nature of the task to the patient by crossing out two of the four lines located in the central column. The patient is then instructed to cross out all the lines they can see on the page. After the patient completes the task, the number of crossed-out lines not in the central column are counted. The maximum score is 36 (18 left, 18 right).

### line_cancellation_template.py description:
A scan of the template for the line cancellation task is read and preprocessed to remove noise and emphasize target lines. The centroid coordinates of relevant detected contours are stored and carefully merged to remove instances of target lines being represented by more than one centroid. The arrow centroid is also isolated and subsequently used to orient the image. The centroids of the four lines in the central column are removed from the final list of centroid coordinates. The constant variable, LineC_T_C1, stores this final list of centroid coordinates (i.e., the centres of each target line) and is passed on to line_cancellation.py.

### line_cancellation.py description:
A scan of the patient's completed line cancellation task is read and heavily preprocessed to emphasize line intersection points. Once again, the centroid coordinates of relevant detected contours are stored and carefully merged to remove instances of line intersection points being represented by more than one centroid. The arrow centroid is again isolated and used to orient the image. The centroid coordinates are then cross-referenced against LineC_T_C1 to determine which target lines were detected. The detected target lines are post-processed to determine the number of crossed-out lines on the left and right sides (LineC_LS, LineC_RS), the total number of crossed-out lines and its corresponding standard value (LineC, LineC_SV), and the resulting horizontal and vertical centres of cancellation (LineC_HCoC, LineC_VCoC).
