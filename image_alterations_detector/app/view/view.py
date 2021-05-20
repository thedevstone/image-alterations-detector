import tkinter as tk
from tkinter.ttk import Notebook

from image_alterations_detector.app.view.tabs.tab1 import Tab1
from image_alterations_detector.app.view.tabs.tab2 import Tab2
from image_alterations_detector.app.view.tabs.tab3 import Tab3
from image_alterations_detector.app.view.toolbar import Toolbar


class View:
    def __init__(self, controller):
        self.controller = controller
        self.window = tk.Tk()
        self.init_view()
        self.toolbar = Toolbar(self, self.window)
        self.tab_control = tk.ttk.Notebook(self.window)
        self.tab1 = Tab1(self, self.tab_control)
        self.tab_control.grid(row=0, column=1, sticky='nsew')
        self.tab2 = Tab2(self, self.tab_control)
        self.tab_control.grid(row=0, column=1, sticky='nsew')
        self.tab3 = Tab3(self, self.tab_control)
        self.tab_control.grid(row=0, column=1, sticky='nsew')

    def init_view(self):
        # Window init
        w, h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        # self.window.attributes('-fullscreen', True)
        self.window.geometry("%dx%d+0+0" % (w, h))
        self.window.title('Image Alteration Detector')
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=0)
        self.window.columnconfigure(1, weight=1)

    def show(self):
        self.window.mainloop()
