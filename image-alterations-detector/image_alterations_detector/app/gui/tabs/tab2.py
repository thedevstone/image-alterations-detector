from __future__ import annotations

import tkinter as tk
from tkinter.ttk import Notebook
from typing import Optional

from image_alterations_detector.app.gui.utils.conversion import convert_to_tk_image
from image_alterations_detector.app.gui.utils.layout_utils import set_img_label_layout


class Tab2:
    def __init__(self, gui, tab_control: Notebook):
        from image_alterations_detector.app.gui.gui import Gui
        # Init tab
        self.gui: Gui = gui
        self.tab_control = tab_control
        self.tab_root = tk.ttk.Frame(self.tab_control)
        # Tab root
        self.tab_root.grid(row=0, column=0, sticky='nsew')
        self.tab_control.add(self.tab_root, text='Analysis')
        self.tab_root.columnconfigure(0, weight=1)
        self.tab_root.rowconfigure(0, weight=1)

        # -------------------- Images panel
        self.images_panel = tk.Frame(self.tab_root)
        self.images_panel.grid(row=0, column=0, sticky='new')
        self.images_panel.columnconfigure(0, weight=1)
        self.images_panel.columnconfigure(1, weight=1)
        # Images
        self.image2_label: Optional[tk.Label] = tk.Label(self.images_panel)
        self.image1_label: Optional[tk.Label] = tk.Label(self.images_panel)

        # -------------------- Info panel
        self.processed_panel = tk.Frame(self.tab_root)
        self.processed_panel.grid(row=1, column=0, sticky='sew')
        self.processed_panel.columnconfigure(0, weight=1)
        self.processed_panel.columnconfigure(1, weight=1)

    def set_triangulation_images(self, img1, img2):
        img1 = convert_to_tk_image(img1)
        img2 = convert_to_tk_image(img2)
        set_img_label_layout(self.image1_label, img1, 0, 0, 'w')
        set_img_label_layout(self.image2_label, img2, 0, 1, 'e')
