from tkinter import Label


def set_img_label_layout(label: Label, img, row, col, sticky):
    label.configure(image=img)
    label.image = img
    label.grid(row=row, column=col, sticky=sticky)
