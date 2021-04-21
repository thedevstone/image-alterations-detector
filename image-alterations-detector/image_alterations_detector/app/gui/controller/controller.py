from typing import Optional

import cv2
import numpy as np

from image_alterations_detector.app.gui.gui import Gui
from image_alterations_detector.app.gui.utils.conversion import image_view_resize_preserve_ratio
from image_alterations_detector.app.gui.utils.general_utils import show_message_box
from image_alterations_detector.dataset.altered_dataset.test_altered_dataset import test_two_images
from image_alterations_detector.face_morphology.landmarks_prediction.visualization import \
    visualize_facial_landmarks_points, visualize_facial_landmarks_areas
from image_alterations_detector.face_transform.face_alignment.face_aligner import FaceAligner


class Controller:
    def __init__(self):
        self.ui: Optional[Gui] = None
        self.img_source: Optional[np.ndarray] = None
        self.img_target: Optional[np.ndarray] = None
        self.face_aligner = FaceAligner()
        self.img_source_aligned: Optional[np.ndarray] = None
        self.img_target_aligned: Optional[np.ndarray] = None
        self.landmarks_source: Optional[np.ndarray] = None
        self.landmarks_target: Optional[np.ndarray] = None

    def check_images(self):
        return self.img_source is None or self.img_target is None

    def load_image_form_path(self, img_path, img_type):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_type == 'source':
            self.img_source = img
            self.ui.tab1.set_image1(image_view_resize_preserve_ratio(self.img_source))
        else:
            self.img_target = img
            self.ui.tab1.set_image2(image_view_resize_preserve_ratio(self.img_target))

    def take_webcam_photo(self):
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cam.release()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.img_source = img
        self.ui.tab1.set_image1(image_view_resize_preserve_ratio(self.img_source))

    def align_images(self):
        if self.check_images():
            show_message_box('please load both images', 'warning')
        else:
            self.img_source_aligned, self.landmarks_source = self.face_aligner.align(self.img_source)
            self.img_target_aligned, self.landmarks_target = self.face_aligner.align(self.img_target)
            # Get alignment view
            visual_source = visualize_facial_landmarks_points(self.img_source_aligned, self.landmarks_source)
            visual_target = visualize_facial_landmarks_points(self.img_target_aligned, self.landmarks_target)
            visual_source = visualize_facial_landmarks_areas(visual_source, self.landmarks_source, alpha=0.5)
            visual_target = visualize_facial_landmarks_areas(visual_target, self.landmarks_target, alpha=0.5)
            self.ui.tab1.show_aligned(visual_source, visual_target)

    def analyze_images(self):
        if self.check_images():
            show_message_box('please load both images', 'warning')
        else:
            res = test_two_images(self.img_source, self.img_target)
            print(res)

    def set_ui(self, ui):
        self.ui = ui

    def start(self):
        self.ui.show()
