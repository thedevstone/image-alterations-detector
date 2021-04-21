import tkinter as tk
from tkinter import filedialog

import cv2

from image_alterations_detector.app.gui.utils.conversion import image_process, image_view_resize_preserve_ratio


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
        btn_align = tk.Button(self.toolbar, text="Align images",
                              command=lambda: self.align_images())
        # Position
        self.toolbar.grid(row=0, column=0, sticky="nsw")
        btn_open_source.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        btn_open_doc.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        btn_open_webcam.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        btn_align.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

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
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = image_process(img)
            self.gui.controller.img_source = img
            self.gui.tab1.set_image1(image_view_resize_preserve_ratio(img, 2))
        elif img_type == 'target':
            img_path = load_file()
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = image_process(img)
            self.gui.controller.img_target = img
            self.gui.tab1.set_image2(image_view_resize_preserve_ratio(img, 2))
        else:
            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            img = image_process(frame)
            cam.release()
            self.gui.controller.img_source = img
            self.gui.tab1.set_image1(image_view_resize_preserve_ratio(img, 2))

    def align_images(self):
        self.gui.controller.align_images()
        self.gui.tab1.show_aligned()
