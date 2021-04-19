from matplotlib import pyplot as plt

from image_alterations_detector.segmentation.test_segmentation import segment_image

if __name__ == '__main__':
    segmented = segment_image('img1.jpg')
    plt.imshow(segmented)
    plt.axis('off')
    plt.show()
