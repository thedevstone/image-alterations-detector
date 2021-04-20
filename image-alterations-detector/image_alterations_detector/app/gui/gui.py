import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Notebook

import cv2

from image_alterations_detector.app.gui.tabs.tab1 import Tab1


class Gui:
    def __init__(self, controller):
        self.controller = controller
        self.window = tk.Tk()
        self.init_gui()
        self.dialog = tk.filedialog
        self.tab_control = tk.ttk.Notebook(self.window)
        self.tab1 = Tab1(self, self.tab_control)
        self.tab_control.pack(expand=1, fill="both")

    def init_gui(self):
        # Window init
        self.window.geometry('800x800')
        self.window.title('Image Alteration Detector')
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)

    def show(self):
        self.window.mainloop()

    def load_image(self, img_type):
        def load_file():
            filepath = self.dialog.askopenfilename(
                filetypes=[("Png files", "*.png"), ("Jpg files", "*.jpg"), ("Jpeg files", "*.jpeg")]
            )
            if not filepath:
                raise FileNotFoundError('Image not found')
            return filepath

        if img_type == 'source':
            img = cv2.imread(load_file(), cv2.IMREAD_COLOR)
            self.controller.img_source = img
        elif img_type == 'target':
            img = cv2.imread(load_file(), cv2.IMREAD_COLOR)
            self.controller.img_target = img
        else:
            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            cam.release()
            self.controller.img_source = frame
