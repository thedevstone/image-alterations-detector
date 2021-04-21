import tkinter as tk
from tkinter.ttk import Notebook

from image_alterations_detector.app.gui.tabs.tab1.tab1 import Tab1
from image_alterations_detector.app.gui.toolbar import Toolbar


class Gui:
    def __init__(self, controller):
        self.controller = controller
        self.window = tk.Tk()
        self.init_gui()
        self.toolbar = Toolbar(self, self.window)
        self.tab_control = tk.ttk.Notebook(self.window)
        self.tab1 = Tab1(self, self.tab_control)
        self.tab_control.grid(row=0, column=1, sticky='nsew')

    def init_gui(self):
        # Window init
        self.window.geometry('1200x800')
        self.window.title('Image Alteration Detector')
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=0)
        self.window.columnconfigure(1, weight=1)

    def show(self):
        self.window.mainloop()
