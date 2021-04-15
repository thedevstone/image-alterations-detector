import numpy as np


def landmarks_to_array(shape, dtype="int"):
    """
    Convert dlib landmark notation to numpy array of tuple

    :param shape: the landmark shape
    :param dtype: the type of output matrix
    :return: the numpy of tuple conversion
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rect_to_bounding_box(rect):
    """
    Take a bounding predicted by dlib and convert it to the format (x, y, w, h) as we would normally do with OpenCV

    :param rect: the bounding rect
    :return: (x, y, w, h) coordinate of the bounding box
    """
    #
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h
