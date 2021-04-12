import matplotlib.pyplot as plt


def get_images_mosaic(title, images_labels_data, rows, cols):
    """
    Construct an image mosaic from a list of tuples

    :param title: the title of the mosaic
    :param images_labels_data: a list of tuples (image, caption)
    :param rows: mosaic rows
    :param cols: mosaic cols
    :return: a matplotlib figure
    :raise: indexError if number of mosaic cells is less than images to show
    """
    total_images = len(images_labels_data)
    figure_mosaic = plt.figure(1, figsize=(20, 10))
    figure_mosaic.suptitle(title, fontsize=30)
    rows = rows
    cols = cols
    if rows * cols < total_images:
        raise IndexError("Number of images grater than number of mosaic cells")
    for idx, image_label_data in enumerate(images_labels_data):
        # Subplot RGB
        image = image_label_data[0]
        caption = image_label_data[1]
        ax_image = figure_mosaic.add_subplot(rows, cols, idx + 1)
        ax_image.axis('off')
        ax_image.set_title(caption, fontsize=20)
        plt.imshow(image)
    return figure_mosaic
