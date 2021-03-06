# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os
import time

from LBP import LocalBinaryPatterns
from face_functions import crop_face_part
from face_functions import FACIAL_LANDMARKS_IDXS
import matplotlib.pyplot as plt

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(8, 1)
data = []
labels = []

# initialize paths to directory

imagesInPath = "demo"
descriptionInPath = "data/description/gender_info.csv"
directory_out = "demo/"
# load description data
# tmp = np.genfromtxt(descriptionInPath, delimiter=', ', dtype=None, encoding=None)
# genderDescDict = {}
# for (N, G) in tmp:
#     if str(G) == "F":
#         genderDescDict[N] = 1
#     else:
#         genderDescDict[N] = 0

N = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# iterate over images
for filename in os.listdir(imagesInPath):
    if filename.endswith("00002_1.ppm") or filename.endswith("00001_1.ppm") or filename.endswith("00031_1.ppm"):
        print(filename)
        N += 1
        start = time.time()
        # load the input image and convert it to grayscale
        image = cv2.imread(imagesInPath + "/" + filename)
        gray = image[:, :, 0]

        # detect faces in the grayscale image
        rects = detector(gray, 1)


        conacHist = []
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the face parts individually
            for (name, (i, j, width, height,ratioW ,ratioH  )) in FACIAL_LANDMARKS_IDXS.items():
                # clone the original
                clone = image.copy()

                # extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

                # crop fragment with proper H - W ratio
                (x, y, w, h) = crop_face_part(x, y, w, h, ratioW, ratioH)
                roi = image[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=width, height=height, inter=cv2.INTER_CUBIC)

                # show the particular face part
                cv2.imshow("ROI "+name, roi)
                cv2.imwrite(directory_out + "ROI_" + name + "_" + filename, roi)

                # describe ROI (LBP)
                hist = desc.describe(roi[:,:,0])

                conacHist = np.concatenate((conacHist, hist), axis=0)
                plt.bar( np.arange(hist.shape[0]), hist)
                plt.title("Histogram" + filename + " of " + name)
                #plt.show()
                plt.savefig(directory_out + "Histogram_" + name + "_" + str(N) + ".png")

            # extract the label from the image path, then update the
            # label and data lists
            # labels.append(genderDescDict[int(filename[0:5])])
            # data.append(conacHist)
            # visualize all facial landmarks
            cv2.imshow("Image", clone)
            end = time.time()

            cv2.waitKey(0)
