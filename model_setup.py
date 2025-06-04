# model_setup.py

import torchvision.models.detection as detection

def get_model_instance_segmentation(num_classes):


    """
    Returns a Mask R-CNN model customized for gas flare detection.
    Args:
        num_classes (int): Number of classes (including background)
    Returns:
        model: Mask R-CNN model
    """
    # Load a pre-trained model on COCO
    model = detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier (box predictor)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


#Loads a pretrained Mask R-CNN
#Modifies it for specific use case: 1 class ("gas flare") + background = 2 class
#Returns the model, ready to train

#use this like this: 
"""
from model_setup import get_model_instance_segmentation
model = get_model_instance_segmentation(num_classes=2)
"""
#Now you have a fully configured model ready to train


