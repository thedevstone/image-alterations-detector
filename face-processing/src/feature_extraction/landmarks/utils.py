# import the necessary packages
from collections import OrderedDict

import cv2
import numpy as np

# Define a dictionary that maps the indexes of the facial landmarks to specific face regions
# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_INDEXES = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_INDEXES = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", 4)
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS_INDEXES = FACIAL_LANDMARKS_68_INDEXES


def visualize_facial_landmarks_points(img, landmarks_2d):
    img_out = np.array(img)
    for n in range(0, 68):
        x = landmarks_2d[n, 0]
        y = landmarks_2d[n, 1]
        cv2.circle(img_out, (x, y), 4, (0, 0, 255), -1)
    return img_out


def visualize_facial_landmarks_areas(image, shape, colors=None, alpha=0.75):
    """
    Visualization of facial landmarks.

    :param image: the input image
    :param shape: the input shape
    :param colors: colors of face regions
    :param alpha: the alpha blending
    :return: the landmark visualization image
    """
    overlay = image.copy()
    output = image.copy()

    # Default coloring
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220), (0, 0, 255)]

    # For each landmark region
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        (init, end) = FACIAL_LANDMARKS_INDEXES[name]
        landmarks = shape[init: end]

        # Since the jawline is a non-enclosed facial region, just draw lines between the (x, y)-coordinates
        if name == "jaw":
            for jaw_point in range(1, len(landmarks)):
                point_a = tuple(landmarks[jaw_point - 1])
                point_b = tuple(landmarks[jaw_point])
                cv2.line(overlay, point_a, point_b, colors[i], 2)
        # Otherwise, compute the convex hull of the facial landmark coordinates points and display it
        else:
            hull = cv2.convexHull(landmarks)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # Apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return output
