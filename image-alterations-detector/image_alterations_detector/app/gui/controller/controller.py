from typing import Optional

import numpy as np

from image_alterations_detector.app.gui.gui import Gui
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

    def align_images(self):
        if self.img_source is None or self.img_target is None:
            raise AssertionError('Images not initialized')
        self.img_source_aligned, self.landmarks_source = self.face_aligner.align(self.img_source)
        self.img_target_aligned, self.landmarks_target = self.face_aligner.align(self.img_target)
        
    def get_alignment_view(self):
        visual_source = visualize_facial_landmarks_points(self.img_source_aligned, self.landmarks_source)
        visual_target = visualize_facial_landmarks_points(self.img_target_aligned, self.landmarks_target)
        visual_source = visualize_facial_landmarks_areas(visual_source, self.landmarks_source, alpha=0.5)
        visual_target = visualize_facial_landmarks_areas(visual_target, self.landmarks_target, alpha=0.5)
        return visual_source, visual_target

    def analyze_images(self):
        if self.img_source is None or self.img_target is None:
            raise AssertionError('Image not initialized')
        res = test_two_images(self.img_source, self.img_target)
        print(res)

    def set_ui(self, ui):
        self.ui = ui

    def start(self):
        self.ui.show()
