import numpy as np


def denormalize(image):
    """
    Denormalize 'float32' image in [0,1] to 'uint8' image in [0,255]
    """
    if image.dtype == 'uint8':
        raise ValueError('Cannot denormalize already uint8 image')
    return (image * 255.).astype('uint8')


def mask_channels_to_rgb(image, n_classes, image_size, colors_values_list):
    """
    Convert denormalized image 'uint8' [0,255] to RGB image [0,255]
    """
    channels = np.dsplit(image, n_classes + 1)
    rgb_out_image = np.zeros((image_size, image_size, 3))
    # Iterate over binary masks and applying color to rgb image only in corresponding foreground pixels in masks
    for idx, color in enumerate(colors_values_list):
        indexing = np.reshape(channels[idx], (image_size, image_size))
        indexing = indexing > 128  # Foreground
        rgb_out_image[indexing] = color
    # applying color to rgb image only in corresponding foreground pixels of background mask
    indexing = np.reshape(channels[-1], (image_size, image_size))
    indexing = indexing > 128
    rgb_out_image[indexing] = [0, 0, 0]
    return rgb_out_image.astype('uint8')
