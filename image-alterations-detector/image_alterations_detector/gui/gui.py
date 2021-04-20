import tkinter as tk
from tkinter.filedialog import askopenfilename


class Gui:
    def __init__(self, ):
        self.window = tk.Tk()
        self.window.geometry('600x600')
        self.window.title('Image Alteration Detector')
        self.window.rowconfigure(0, minsize=500, weight=1)
        self.window.columnconfigure(0, minsize=500, weight=1)

        fr_buttons = tk.Frame(self.window, relief=tk.RAISED, bd=2)
        fr_buttons.grid(row=0, sticky="ew")
        fr_buttons.grid(row=2, sticky="ew")

        btn_open = tk.Button(fr_buttons, text="Open", command=self.open_file)

        btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        fr_buttons.grid(row=0, column=0, sticky="ns")

    def show(self):
        self.window.mainloop()

    def open_file():
        """Open a file for editing."""
        filepath = askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not filepath:
            return


if __name__ == '__main__':
    gui = Gui()
    gui.show()
