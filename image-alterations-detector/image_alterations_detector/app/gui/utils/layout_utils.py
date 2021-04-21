from tkinter import Label, StringVar


def set_img_label_layout(label: Label, img, row, col, sticky):
    label.configure(image=img)
    label.image = img
    label.grid(row=row, column=col, sticky=sticky)


def create_text_label(parent, text: str, font_size, row, col, sticky):
    label = Label(parent, text=text, font=('Courier', font_size))
    label.grid(row=row, column=col, sticky=sticky)
    return label


def create_text_label_var(parent, text_var: StringVar, font_size, row, col, sticky):
    label = Label(parent, textvariable=text_var, font=('Courier', font_size))
    label.grid(row=row, column=col, sticky=sticky)
    return label
