from tkinter import messagebox


def show_message_box(message, msg_type):
    if msg_type == 'warning':
        messagebox.showwarning(message=message)
    elif msg_type == 'error':
        messagebox.showerror(message=message)
    elif msg_type == 'info':
        messagebox.showinfo(message=message)
