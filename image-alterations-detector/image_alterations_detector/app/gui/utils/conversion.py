import cv2
from PIL import Image, ImageTk


def convert_to_tk_image(img):
    # convert the images to PIL format...
    img = Image.fromarray(img)
    # ...and then to ImageTk format
    img = ImageTk.PhotoImage(img)
    return img


def image_view_resize_preserve_ratio(img, ratio=2):
    img = cv2.resize(img, (int(img.shape[1] / ratio), int(img.shape[0] / ratio)))
    return img


def image_view_resize(img, size=512):
    return cv2.resize(img, (size, size))
