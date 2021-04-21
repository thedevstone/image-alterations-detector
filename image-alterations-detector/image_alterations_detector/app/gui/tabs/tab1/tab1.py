from __future__ import annotations

import tkinter as tk
from tkinter import VERTICAL
from tkinter.ttk import Notebook
from typing import Optional

from image_alterations_detector.app.gui.utils.conversion import convert_to_tk_image


class Tab1:
    def __init__(self, gui, tab_control: Notebook):
        from image_alterations_detector.app.gui.gui import Gui
        # Init tab
        self.gui: Gui = gui
        self.tab_control = tab_control
        self.tab_root = tk.ttk.Frame(self.tab_control)
        # Tab root
        self.tab_root.grid(row=0, column=0, sticky='nsew')
        self.tab_control.add(self.tab_root, text='Loading')
        # Init panels
        self.images_panel = tk.PanedWindow(self.tab_root, orient=VERTICAL)
        self.processed_pane = tk.PanedWindow(self.tab_root, orient=VERTICAL)
        self.images_panel.grid(row=0, column=0, sticky='new')
        # Images
        self.image1_panel: Optional[tk.Label] = tk.Label(self.images_panel)
        self.image2_panel: Optional[tk.Label] = tk.Label(self.images_panel)

    def set_image1(self, img1):
        img1 = convert_to_tk_image(img1)
        self.image1_panel.configure(image=img1)
        self.image1_panel.image = img1
        self.image1_panel.grid(row=0, column=0, sticky='nsew')

    def set_image2(self, img2):
        img2 = convert_to_tk_image(img2)
        self.image2_panel.configure(image=img2)
        self.image2_panel.image = img2
        self.image2_panel.grid(row=0, column=1, sticky='nsew')
