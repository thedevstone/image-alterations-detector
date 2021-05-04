from image_alterations_detector.app.utils.conversion import image_resize_with_border
from image_alterations_detector.plotting.plotting import get_images_mosaic_with_label
from image_alterations_detector.segmentation.face_segmenter import denormalize_and_convert_rgb, FaceSegmenter
from image_alterations_detector.utils.image_utils import load_img

if __name__ == '__main__':
    face_segmenter = FaceSegmenter()
    segmented = face_segmenter.segment_images(
        [image_resize_with_border(load_img('franco1.png')),
         image_resize_with_border(load_img('franco2.png')),
         image_resize_with_border(load_img('franco3.png')),
         image_resize_with_border(load_img('franco4.bmp'))
         ])
    rgb_images = denormalize_and_convert_rgb(segmented)
    get_images_mosaic_with_label('Segmented',
                                 [(rgb_images[0], 'img1'),
                                  (rgb_images[1], 'img2'),
                                  (rgb_images[2], 'img2')], 1,
                                 3).show()
    # print('IOU:', compute_general_iou(segmented[0], segmented[1]))

    # print('Single IOU:', compute_iou_per_mask(segmented[0], segmented[1]))
