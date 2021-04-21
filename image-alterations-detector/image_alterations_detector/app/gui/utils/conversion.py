import numpy as np
from PIL import Image, ImageTk
from imutils import resize


def convert_to_tk_image(img):
    # convert the images to PIL format...
    img = Image.fromarray(img)
    # ...and then to ImageTk format
    img = ImageTk.PhotoImage(img)
    return img


def image_view_resize(img, size=512):
    img = resize(img, width=size) if img.shape[0] > img.shape[1] else resize(img, height=size)
    center = np.array(img.shape) / 2
    x = center[1] - size / 2
    y = center[0] - size / 2
    x = int(x)
    y = int(y)

    crop_img = img[y:y + size, x:x + size]
    return crop_img


def mean_weight(result_tuple, svm_rm_weight=1.2, mlp_weight=1.0):
    svm_rf = result_tuple[1] * svm_rm_weight
    mlp = result_tuple[2] * mlp_weight
    result = (svm_rf + mlp) / (svm_rm_weight + mlp_weight)
    result = round(result, 2)
    result *= 100.0
    return result
