# import the necessary packages
import cv2
import numpy as np

from feature_extraction.landmarks.utils import FACIAL_LANDMARKS_68_INDEXES, FACIAL_LANDMARKS_5_INDEXES


class FaceAligner:
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=256, desired_face_height=None):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height
        if self.desired_face_height is None:
            self.desired_face_height = self.desired_face_width

    def align(self, image, shape):
        # Extract the left and right eye (x, y)-coordinates
        if len(shape) == 68:
            (lStart, lEnd) = FACIAL_LANDMARKS_68_INDEXES["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_INDEXES["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_INDEXES["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_INDEXES["right_eye"]
        # Get eye points
        left_eye_pts = shape[lStart:lEnd]
        right_eye_pts = shape[rStart:rEnd]
        # Compute the center of mass for each eye
        left_eye_center = left_eye_pts.mean(axis=0).astype("int")
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")
        # Compute the angle between the eye centroids
        d_y = right_eye_center[1] - left_eye_center[1]
        d_x = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(d_y, d_x)) - 180
        # Compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((d_x ** 2) + (d_y ** 2))
        desiredDist = (desired_right_eye_x - self.desired_left_eye[0])
        desiredDist *= self.desired_face_width
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((left_eye_center[0] + right_eye_center[0]) // 2,
                      (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output
