from image_alterations_detector.dataset.altered_dataset.altered_dataset_utils import load_altered_dataset
from image_alterations_detector.dataset.altered_dataset.test_altered_dataset import test_two_images, \
    test_on_extra_dataset_images

if __name__ == '__main__':
    dataset_path = '/Users/luca/Desktop/altered'
    genuine, beauty = load_altered_dataset(dataset_path)
    image_index = 3
    print('Test intra dataset')
    genuine_img = genuine[0][image_index]
    controlled_Img = genuine[2][image_index]
    beauty_img = beauty[2][image_index]

    print('Genuine -> Controlled', test_two_images(genuine_img, controlled_Img))
    print('Genuine -> Beauty', test_two_images(genuine_img, beauty_img))

    print('Test extra dataset')
    print('Genuine -> Genuine', test_on_extra_dataset_images('test_luca1.jpg', 'test_luca1.jpg'))
    print('Genuine -> Genuine liquified', test_on_extra_dataset_images('test_luca1.jpg', 'test_luca1_liquify.png'))
    print('Genuine -> Different pose', test_on_extra_dataset_images('test_luca1.jpg', 'test_luca2.jpg'))
    print('Genuine -> Beauty', test_on_extra_dataset_images('test_luca1.jpg', 'test_luca1_beauty.jpeg'))
