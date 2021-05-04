from image_alterations_detector.plotting.plotting import get_images_mosaic_with_label
from image_alterations_detector.segmentation.face_segmenter import FaceSegmenter, denormalize_and_convert_rgb
from image_alterations_detector.utils.image_utils import load_img

if __name__ == '__main__':
    face_segmenter = FaceSegmenter()
    segmented = face_segmenter.segment_images_keep_aspect_ratio(
        [load_img('franco1.png'),
         load_img('franco2.png'),
         load_img('franco3.png'),
         load_img('franco4.bmp')
         ])
    rgb_images = denormalize_and_convert_rgb(segmented)
    get_images_mosaic_with_label('Segmented',
                                 [(rgb_images[0], 'img1'),
                                  (rgb_images[1], 'img2'),
                                  (rgb_images[2], 'img3'),
                                  (rgb_images[3], 'img4')
                                  ], 2, 2).show()
    # print('IOU:', compute_general_iou(segmented[0], segmented[1]))
    #
    # print('Single IOU:', compute_iou_per_mask(segmented[0], segmented[1]))
