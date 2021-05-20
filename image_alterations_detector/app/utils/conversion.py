import cv2
import numpy as np
from PIL import Image, ImageTk
from imutils import resize


def convert_to_tk_image(img):
    # convert the images to PIL format...
    img = Image.fromarray(img)
    # ...and then to ImageTk format
    img = ImageTk.PhotoImage(img)
    return img


def image_resize_with_border(img, size=512):
    old_size = img.shape[:2]
    ratio = float(size) / max(old_size)
    new_size = tuple([round(x * ratio) for x in old_size])
    im = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    mid_border_y = delta_h // 2
    mid_border_x = delta_w // 2
    top, bottom = mid_border_y, delta_h - mid_border_y
    left, right = mid_border_x, delta_w - mid_border_x
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img, old_size, (top, bottom, left, right)


def image_resize_restore_ratio(img, new_size, border):
    crop_img = img[border[0]: img.shape[0] - border[1], border[2]: img.shape[1] - border[3]]
    old_size = crop_img.shape[:2]
    restored_img = cv2.resize(crop_img, (new_size[1], new_size[0]))
    return restored_img


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
    result *= 100.0
    result = round(result, 2)
    return result


def weighted_majority_mean(results):
    majority = np.where(results >= 50.0)
    weight_majority = len(majority[0])
    minority = np.where(results < 50.0)
    weight_minority = len(minority[0])
    majority_mean = results[majority[0]].mean() if len(majority[0] > 0) else 0
    minority_mean = results[minority[0]].mean() if len(minority[0] > 0) else 0
    final_estimate = (majority_mean * weight_majority + minority_mean * weight_minority) / (
            weight_majority + weight_minority)
    final_estimate = round(final_estimate, 2)
    return final_estimate


def weighted_mean_accuracy(results, accuracies):
    mutliply = results * accuracies
    mean = mutliply.sum() / accuracies.sum()
    return round(mean, 2)


if __name__ == '__main__':
    result = np.array([30.0, 30.0, 40.0, 10.0])
    # res = weighted_majority_mean(result)
    # print(res)
    acc = np.array(
        [mean_weight(['Angles', 0.60, 0.56]), mean_weight(['Areas', 0.69, 0.70]),
         mean_weight(['Centroids', 0.68, 0.71]),
         mean_weight(['Matrices', 0.70, 0.67])])
    res = weighted_mean_accuracy(result, acc)
    print(res)
