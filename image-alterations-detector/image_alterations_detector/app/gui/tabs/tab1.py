from __future__ import annotations

import tkinter as tk
from tkinter.ttk import Notebook


class Tab1:
    def __init__(self, gui, tab_control: Notebook):
        from image_alterations_detector.app.gui.gui import Gui
        # Init tab
        self.gui: Gui = gui
        self.tab_control = tab_control
        self.tab_root = tk.ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_root, text='Loading')
        # Init children
        self.toolbar = Toolbar(self.gui, self.tab_root)


class Toolbar:
    def __init__(self, gui, tab_root):
        from image_alterations_detector.app.gui.gui import Gui
        self.gui: Gui = gui
        self.tab_root = tab_root
        self.toolbar = tk.Frame(self.tab_root, relief=tk.RAISED, bd=2)
        btn_open_source = tk.Button(self.toolbar, text="Load source image",
                                    command=lambda: self.gui.load_image('source'))
        btn_open_doc = tk.Button(self.toolbar, text="Load target image",
                                 command=lambda: self.gui.load_image('target'))
        btn_open_webcam = tk.Button(self.toolbar, text="Take a photo",
                                    command=lambda: self.gui.load_image('webcam'))
        self.toolbar.grid(row=0, column=0, sticky="new")
        btn_open_source.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        btn_open_doc.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        btn_open_webcam.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
