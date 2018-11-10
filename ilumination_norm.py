import cv2
import numpy as np
# import dlib
import os
import csv


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)



directory_in = "data/images_gray"
directory_out = "data/images_norm/"


imagelist=[]
mean_list=[]



with open("ilu_data.csv", "r") as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        imagelist.append(row[1])
        mean_list.append(float(row[0]))

print(imagelist)
print(mean_list)
I_MEAN=np.mean(mean_list)
I_STD=np.std(mean_list)
print(I_MEAN)
print(I_STD)

gamma_list=[]
for elem in  mean_list:
    gamma_list.append(I_MEAN/elem)



# iterate directory with data and rename files
for i, filename in enumerate(imagelist):
    # load the input image
    image = cv2.imread(directory_in + "/" + filename[1:])

    # apply gamma correction and show the images
    adjusted = adjust_gamma(image, gamma=gamma_list[i])
    cv2.imwrite(directory_out + filename[1:], adjusted)





# # iterate directory with data and rename files
# for filename in os.listdir(directory_in):
#     if filename.endswith(".ppm"):
#         # load the input image
#         image = cv2.imread(directory_in + "/" + filename)
#
#
#         imagelist.append(filename)
#         mean_list.append(np.mean(image))
#
# print(imagelist)
# print(mean_list)
# I_MEAN=np.mean(mean_list)
# I_STD=np.std(mean_list)
# print(I_MEAN)
# print(I_STD)
# new_file =""
# for i, elem in enumerate(mean_list):
#     new_file +=  str(elem) + ', ' + imagelist[i] + '\n'
# fid = open("ilu_data.csv", 'w')
# fid.write(new_file)
# fid.close()
