from typing import Optional

import numpy as np

from image_alterations_detector.app.gui.gui import Gui
from image_alterations_detector.dataset.altered_dataset.test_altered_dataset import test_two_images


class Controller:
    def __init__(self):
        self.ui: Optional[Gui] = None
        self.img_source: Optional[np.ndarray] = None
        self.img_target: Optional[np.ndarray] = None

    def test_two_images(self):
        if not (self.img_source and self.img_target):
            raise AssertionError('Image not initialized')
        res = test_two_images(self.img_source, self.img_target)
        return res

    def set_ui(self, ui):
        self.ui = ui

    def start(self):
        self.ui.show()
