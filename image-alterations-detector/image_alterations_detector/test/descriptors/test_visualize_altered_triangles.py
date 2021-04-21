import cv2

from image_alterations_detector.descriptors.double_image_alteration_descriptors.triangle_descriptor_visualization import \
    draw_delaunay_alterations
from image_alterations_detector.utils.image_utils import load_img


def main():
    # dataset_path = '/Users/luca/Desktop/altered'
    # genuine, beauty = load_altered_dataset(dataset_path)
    # image_index = 3
    # print('Test intra dataset')
    # genuine_img = genuine[0][image_index]
    # controlled_Img = genuine[2][image_index]
    # beauty_img = beauty[2][image_index]

    def show_img(img):
        cv2.imshow('Delaunay', img)

    draw_delaunay_alterations(load_img('m-001-1.png'), load_img('m-001-a.jpg'), animate=True, show_function=show_img)


if __name__ == '__main__':
    main()
