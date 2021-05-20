import numpy as np
from skimage import feature


class LocalBinaryPattern:
    def __init__(self, num_points, radius):
        """ Initialize lbp feature extractor with parameters

        :param num_points: number of points in the circle
        :param radius: the radius of the circle
        """
        # store the number of points and radius
        self.numPoints = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        """ Compute the Local Binary Pattern representation of the image, and then use the LBP representation to build the histogram of patterns

        :param image: the image
        :param eps: the precision
        :return: the histogram
        """
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist
