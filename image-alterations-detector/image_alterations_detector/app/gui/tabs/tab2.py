from __future__ import annotations

import tkinter as tk
from tkinter import StringVar
from tkinter.ttk import Notebook
from typing import Optional

from image_alterations_detector.app.gui.utils.conversion import convert_to_tk_image, mean_weight, image_view_resize
from image_alterations_detector.app.gui.utils.layout_utils import set_img_label_layout, create_text_label, \
    create_text_label_var


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
        self.tab_root.rowconfigure(1, weight=0)

        # -------------------- Images panel
        self.images_panel = tk.Frame(self.tab_root)
        self.images_panel.grid(row=0, column=0, sticky='new')
        self.images_panel.columnconfigure(0, weight=1)
        self.images_panel.columnconfigure(1, weight=1)
        # Images
        self.image2_label: Optional[tk.Label] = tk.Label(self.images_panel)
        self.image1_label: Optional[tk.Label] = tk.Label(self.images_panel)

        # -------------------- Info panel
        self.info_panel = tk.Frame(self.tab_root, relief=tk.RAISED, bd=1)
        self.info_panel.grid(row=1, column=0, sticky='nsew')
        self.info_panel.columnconfigure(0, weight=1)

        # Title
        self.title_label: Optional[tk.Label] = tk.Label(self.info_panel, text='Alteration Estimate', font=("Times", 44))
        self.title_label.grid(row=0, column=0, sticky='new')

        # Result panel
        self.result_panel = tk.Frame(self.info_panel, relief=tk.RAISED, bd=1)
        self.result_panel.grid(row=1, column=0, sticky='nsew')
        self.result_panel.columnconfigure(0, weight=1)
        self.result_panel.columnconfigure(1, weight=1)

        create_text_label(self.result_panel, 'Triangles transformation:', 30, 1, 0, 'nw')
        create_text_label(self.result_panel, 'Angles change:', 30, 2, 0, 'nw')
        create_text_label(self.result_panel, 'Texture change:', 30, 3, 0, 'nw')
        # Vars
        self.matrices_var = StringVar()
        create_text_label_var(self.result_panel, self.matrices_var, 30, 1, 1, 'nw')
        self.angles_var = StringVar()
        create_text_label_var(self.result_panel, self.angles_var, 30, 2, 1, 'nw')
        self.texture_var = StringVar()
        create_text_label_var(self.result_panel, self.texture_var, 30, 3, 1, 'nw')

    def set_triangulation_images(self, img1, img2):
        size = int(self.tab_root.winfo_height() - 100)
        img1 = convert_to_tk_image(image_view_resize(img1, size))
        img2 = convert_to_tk_image(image_view_resize(img2, size))
        set_img_label_layout(self.image1_label, img1, 0, 0, 'w')
        set_img_label_layout(self.image2_label, img2, 0, 1, 'e')

    def show_result(self, results):
        angles_result, affine_matrices_result, lbp_result = results
        self.angles_var.set('{}%'.format(mean_weight(angles_result)))
        self.matrices_var.set('{}%'.format(mean_weight(affine_matrices_result)))
        self.texture_var.set('{}%'.format(mean_weight(lbp_result)))
