import tkinter as tk
from tkinter import StringVar
from tkinter.ttk import Notebook
from typing import Optional

import numpy as np

from image_alterations_detector.app.utils.conversion import convert_to_tk_image, mean_weight, image_resize_with_border, \
    weighted_mean_accuracy
from image_alterations_detector.app.utils.layout_utils import set_img_label_layout, create_text_label, \
    create_text_label_var


class Tab2:
    def __init__(self, view, tab_control: Notebook):
        from image_alterations_detector.app.view.view import View
        # Init tab
        self.view: View = view
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
        self.title_label: Optional[tk.Label] = tk.Label(self.info_panel, text='Alteration Estimate', font=("Times", 35))
        self.title_label.grid(row=0, column=0, sticky='new')

        # Result panel
        self.result_panel = tk.Frame(self.info_panel, relief=tk.RAISED, bd=1)
        self.result_panel.grid(row=1, column=0, sticky='nsew')
        self.result_panel.columnconfigure(0, weight=1)
        self.result_panel.columnconfigure(1, weight=1)

        create_text_label(self.result_panel, 'Triangles angles change:', 25, 0, 0, 'nw')
        create_text_label(self.result_panel, 'Triangles areas change:', 25, 1, 0, 'nw')
        create_text_label(self.result_panel, 'Triangles centroids change:', 25, 2, 0, 'nw')
        create_text_label(self.result_panel, 'Triangles affine transformation:', 25, 3, 0, 'nw')
        create_text_label(self.result_panel, 'Distorted:', 35, 4, 0, 'nw')
        create_text_label(self.result_panel, 'Beautified:', 35, 5, 0, 'nw')
        # Vars
        self.angles_var = StringVar()
        self.areas_var = StringVar()
        self.centroids_var = StringVar()
        self.matrices_var = StringVar()
        self.distorted_var = StringVar()
        self.texture_var = StringVar()
        create_text_label_var(self.result_panel, self.angles_var, 25, 0, 1, 'nw')
        create_text_label_var(self.result_panel, self.areas_var, 25, 1, 1, 'nw')
        create_text_label_var(self.result_panel, self.centroids_var, 25, 2, 1, 'nw')
        create_text_label_var(self.result_panel, self.matrices_var, 25, 3, 1, 'nw')
        create_text_label_var(self.result_panel, self.distorted_var, 35, 4, 1, 'nw')
        create_text_label_var(self.result_panel, self.texture_var, 35, 5, 1, 'nw')

    def set_triangulation_images(self, img1, img2):
        size = int(self.tab_root.winfo_height() - 200)
        img1 = convert_to_tk_image(image_resize_with_border(img1, size)[0])
        img2 = convert_to_tk_image(image_resize_with_border(img2, size)[0])
        set_img_label_layout(self.image1_label, img1, 0, 0, 'w')
        set_img_label_layout(self.image2_label, img2, 0, 1, 'e')

    def show_result(self, results):
        angles_result, areas_result, centroids_result, affine_matrices_result, lbp_result = results
        angles_result = mean_weight(angles_result)
        areas_result = mean_weight(areas_result)
        centroids_result = mean_weight(centroids_result)
        affine_matrices_result = mean_weight(affine_matrices_result)
        rounded_results = np.array([angles_result, areas_result, centroids_result, affine_matrices_result])
        accuracies = np.array(
            [mean_weight(['Angles', 0.60, 0.56]), mean_weight(['Areas', 0.69, 0.70]),
             mean_weight(['Centroids', 0.68, 0.71]),
             mean_weight(['Matrices', 0.70, 0.67])])
        mean_distortion = weighted_mean_accuracy(rounded_results, accuracies)

        self.angles_var.set('{}%'.format(angles_result))
        self.areas_var.set('{}%'.format(areas_result))
        self.centroids_var.set('{}%'.format(centroids_result))
        self.matrices_var.set('{}%'.format(affine_matrices_result))
        self.distorted_var.set('{}%'.format(mean_distortion))
        self.texture_var.set('{}%'.format(mean_weight(lbp_result)))
