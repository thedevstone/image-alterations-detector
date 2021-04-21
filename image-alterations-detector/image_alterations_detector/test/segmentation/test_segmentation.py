from image_alterations_detector.plotting.plotting import get_images_mosaic_with_label
from image_alterations_detector.segmentation.segmentation_tools import compute_iou_per_mask, segment_images, \
    compute_general_iou, denormalize_and_convert_rgb
from image_alterations_detector.utils.image_utils import load_img

if __name__ == '__main__':
    segmented = segment_images([load_img('img1.jpg'), load_img('img2.jpg')])
    rgb_images = denormalize_and_convert_rgb(segmented)
    get_images_mosaic_with_label('IOU', [(rgb_images[0], 'img1'), (rgb_images[1], 'img2')], 1, 2).show()
    print('IOU:', compute_general_iou(segmented[0], segmented[1]))

    print('Single IOU:', compute_iou_per_mask(segmented[0], segmented[1]))
