import segmentation_models as sm
from tensorflow import keras


def load_model(model_path):
    dice_loss = sm.losses.DiceLoss()
    jackard_loss = sm.losses.JaccardLoss()
    total_loss = dice_loss + jackard_loss
    inference_model = keras.models.load_model(model_path, custom_objects={'dice_loss_plus_jaccard_loss': total_loss,
                                                                          'iou_score': sm.metrics.IOUScore(
                                                                              threshold=0.7),
                                                                          'f1-score': sm.metrics.FScore(
                                                                              threshold=0.7)})
    return inference_model
