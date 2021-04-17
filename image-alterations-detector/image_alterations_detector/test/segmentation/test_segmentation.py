from image_alterations_detector.face_transform.face_alignment.face_aligner import FaceAligner
from image_alterations_detector.file_system.path_utilities import get_image_path, get_model_path
from image_alterations_detector.segmentation.configuration.keras_backend import set_keras_backend


def main():
    set_keras_backend()
    import cv2
    import matplotlib.pyplot as plt
    import image_alterations_detector.segmentation.configuration.color_configuration as color_conf
    import image_alterations_detector.segmentation.conversions as conversions
    import image_alterations_detector.segmentation.model as model

    # Configuration
    image_size = 256
    aligner = FaceAligner(desired_face_width=image_size)
    classes_to_segment = {'skin': True, 'nose': True, 'eye': True, 'brow': True, 'ear': True, 'mouth': True,
                          'hair': True, 'neck': True, 'cloth': False}
    classes_list, colors_values_list = color_conf.get_classes_colors(classes_to_segment)

    # Images
    img1 = cv2.imread(get_image_path('img1.jpg'), cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1, landmarks = aligner.align(img1)
    img1 = img1.reshape((1, img1.shape[0], img1.shape[1], img1.shape[2])).astype('float')
    img1_normalized = img1 / 255.0

    # Predict
    inference_model = model.load_model(get_model_path('unet.h5'))

    images_predicted = inference_model.predict_one(img1_normalized)
    predicted = conversions.denormalize(images_predicted[0])
    predicted_rgb = conversions.mask_channels_to_rgb(predicted, 8, image_size, colors_values_list)
    plt.imshow(predicted_rgb)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
