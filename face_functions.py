import numpy as np
import collections



# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = collections.OrderedDict([
	("eyebrows", (17, 27, 250, 50, 10, 2)),
	("eyes", (36, 48, 250, 50, 10, 2)),
	("nose", (27, 36, 100, 100, 2, 2)),
    ("mouth", (48, 68, 150, 50, 6, 2)),
	("jaw", (5, 12, 150, 50, 6, 2))
])


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


