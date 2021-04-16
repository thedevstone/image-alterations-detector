import image_alterations_detector.segmentation.configuration.color_configuration as configuration

if __name__ == '__main__':
    classes_to_segment = {'skin': True, 'nose': True, 'eye': True, 'brow': True, 'ear': True, 'mouth': True,
                          'hair': True, 'neck': True, 'cloth': False}
    figure_color = configuration.visualize_color_configuration(classes_to_segment)
    figure_color.show()
