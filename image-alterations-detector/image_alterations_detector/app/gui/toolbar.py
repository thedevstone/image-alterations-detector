import tkinter as tk
from tkinter import filedialog

import cv2

from image_alterations_detector.app.gui.utils.conversion import image_process


class Toolbar:
    def __init__(self, gui, parent):
        from image_alterations_detector.app.gui.gui import Gui
        self.gui: Gui = gui
        self.tab_root = parent
        self.dialog = tk.filedialog
        self.toolbar = tk.Frame(self.tab_root, relief=tk.RAISED, bd=2)
        btn_open_source = tk.Button(self.toolbar, text="Load source image",
                                    command=lambda: self.load_image('source'))
        btn_open_doc = tk.Button(self.toolbar, text="Load target image",
                                 command=lambda: self.load_image('target'))
        btn_open_webcam = tk.Button(self.toolbar, text="Take a photo",
                                    command=lambda: self.load_image('webcam'))
        # Position
        self.toolbar.grid(row=0, column=0, sticky="nsw")
        btn_open_source.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        btn_open_doc.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        btn_open_webcam.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

    def load_image(self, img_type):
        def load_file():
            filepath = self.dialog.askopenfilename(
                filetypes=[("Png files", "*.png"), ("Jpg files", "*.jpg"), ("Jpeg files", "*.jpeg")]
            )
            if not filepath:
                raise FileNotFoundError('Image not found')
            return filepath

        if img_type == 'source':
            img_path = load_file()
            img = image_process(img_path)
            self.gui.controller.img_source = img
            self.gui.tab1.set_image1(self.gui.controller.img_source)
        elif img_type == 'target':
            img_path = load_file()
            img = image_process(img_path)
            self.gui.controller.img_target = img
            self.gui.tab1.set_image2(self.gui.controller.img_target)
        else:
            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            img = image_process(frame)
            cam.release()
            self.gui.controller.img_source = img
