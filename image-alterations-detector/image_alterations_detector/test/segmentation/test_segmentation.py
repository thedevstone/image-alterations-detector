from image_alterations_detector.segmentation.segmentation_tools import compute_iou_per_mask
from image_alterations_detector.utils.image_utils import load_img

if __name__ == '__main__':
    # segmented = segment_images([load_img('img1.jpg'), load_img('img2.jpg')], convert_to_rgb=True)
    # get_images_mosaic_with_label('IOU', [(segmented[0], 'img1'), (segmented[1], 'img2')], 1, 2).show()
    # print('IOU:', compute_general_iou(load_img('img1.jpg'), load_img('img2.jpg')))

    compute_iou_per_mask(load_img('img1.jpg'), load_img('img2.jpg'))
