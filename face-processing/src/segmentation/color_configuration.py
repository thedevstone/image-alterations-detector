from itertools import compress


def get_colors(classes_to_segment):
    all_colors = {'blue': [0, 0, 204], 'green': [0, 153, 76], 'water': [0, 204, 204], 'orange': [255, 51, 51],
                  'purple': [204, 0, 204],
                  'yellow': [255, 255, 0], 'lilla': [204, 204, 255], 'dark_blue': [0, 51, 102], 'blue2': [0, 0, 255],
                  'light_green': [0, 204, 102], 'light_blue': [0, 255, 255], 'red': [204, 0, 0],
                  'violet': [153, 51, 255],
                  'dark_green': [0, 60, 0], 'brown': [150, 75, 0]}

    class_color_mapping = {'skin': 'blue', 'nose': 'green', 'eye': 'violet', 'brow': 'brown', 'ear': 'yellow',
                           'mouth': 'red',
                           'hair': 'orange', 'neck': 'light_blue', 'cloth': 'purple'}

    classes_to_segment_boolean_indexing = list(classes_to_segment.values())

    classes_list = list(compress(classes_to_segment, classes_to_segment_boolean_indexing))
    colors_list = [class_color_mapping.get(key) for key in classes_list]
    colors_values_list = [all_colors.get(key) for key in colors_list]

    return colors_values_list
    
    # figure_colors = plt.figure(figsize=(20, 4))
    # plt.suptitle("Class to color", fontsize=30)
    # 
    # for idx, elem in enumerate(zip(classes_list, colors_values_list)):
    #     plt.subplot(1, len(classes_list), idx + 1)
    #     plt.title('{}'.format(elem[0]), fontsize=20)
    #     plt.imshow(np.full((50, 50, 3), elem[1], dtype='uint8'))
    #     plt.axis('off')
