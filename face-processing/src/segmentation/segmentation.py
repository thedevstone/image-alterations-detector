import os

os.environ['SM_FRAMEWORK'] = 'tf.keras'

import cv2

import color_configuration as color_conf
import model
import matplotlib.pyplot as plt
import conversions

# Configuration
delete_run = True
project_name = 'FaceParsingUNet'
artifact_model_iou = 0.86
artifact_model_architecture = 'UNet'
artifact_model_backbone = 'efficientnetb3'
image_size = 512
classes_to_segment = {'skin': True, 'nose': True, 'eye': True, 'brow': True, 'ear': True, 'mouth': True,
                      'hair': True, 'neck': True, 'cloth': False}
colors_values_list = color_conf.get_colors(classes_to_segment)
n_classes = len(colors_values_list)

# Images
img1 = cv2.imread('../../images/img1.jpg', cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (image_size, image_size))
img1 = img1.reshape(1, img1.shape[0], img1.shape[1], img1.shape[2]).astype('float')
img1_normalized = img1 / 255.0

# Predict
inference_model = model.load_model('../../models/unet.h5')

images_predicted = inference_model.predict(img1_normalized)
predicted = conversions.denormalize(images_predicted[0])
predicted_rgb = conversions.mask_channels_to_rgb(predicted, 8, image_size, colors_values_list)
plt.imshow(predicted_rgb)
plt.show()
