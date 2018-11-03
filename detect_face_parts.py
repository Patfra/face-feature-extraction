# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import collections
import time




def crop_face_part(x, y, w, h, ratioW, ratioH):
    n_w = w * 1
    n_h = h * 1

    if (w / h < ratioW / ratioH):
        while(n_h % ratioH != 0 ):
            n_h = n_h + 1
        n_w = int(n_h  * ratioW / ratioH)
    else:
        if (w / h > ratioW / ratioH):
            while(n_w % ratioW != 0 ):
                n_w = n_w + 1
            n_h = int(n_w  * ratioH / ratioW)
        else:
            n_w = w * 1
            n_h = h * 1
    n_x = int(x - np.floor_divide(n_w - w, 2))
    n_y = int(y - np.floor_divide(n_h - h, 2))

    return n_x, n_y, n_w, n_h


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = collections.OrderedDict([
	("eyebrows", (17, 27, 250, 50, 10, 2)),
	("eyes", (36, 48, 250, 50, 10, 2)),
	("nose", (27, 36, 100, 100, 2, 2)),
    ("mouth", (48, 68, 150, 50, 6, 2)),
	("jaw", (5, 12, 150, 50, 6, 2))
])


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


start = time.time()
# load the input image, resize it, and convert it to grayscale
image = cv2.imread("00001_2.ppm")
gray = image[:, :, 0]

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the face parts individually
    for (name, (i, j, width, height,ratioW ,ratioH  )) in FACIAL_LANDMARKS_IDXS.items():
    # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = image.copy()

        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

        # crop fragment with proper H - W ratio
        (x, y, w, h) = crop_face_part(x, y, w, h, ratioW, ratioH)
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=width, height=height, inter=cv2.INTER_CUBIC)
        print(roi.shape, " ", name)

        # show the particular face part
        cv2.imshow("ROI "+name, roi)

    # visualize all facial landmarks
    cv2.imshow("Image", clone)
    end = time.time()
    print(end - start)
    cv2.waitKey(0)
