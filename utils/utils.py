import json
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize

IMAGENET_NORMALIZE = Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
IMAGENET_DENORMALIZE = Compose([
    Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
    Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
])

with open('safety/utils/imagenet_labels.json', 'r') as f:
    IMAGENET_LABELS = json.load(f)


def load_example_image(preprocess=True):
    """Loads an example image of a Panda and its label.

    Args:
        preprocess (bool): If true, applies imagenet preprocessing.
    """
    image = np.load('safety/utils/images/panda.npy')
    label = 388
    if preprocess:
        image = image / 255
        image = torch.Tensor(image)
        image = IMAGENET_NORMALIZE(image)
    return image, label


def make_single_prediction(model, image):
    """Makes a prediction on a single image

    Args:
        model (nn.Module): the prediction network
        image (torch.Tensor): the input image
    """
    pred = model.forward(image.unsqueeze(0))
    pred = torch.softmax(pred)[0]
    pred_index = pred.argmax().item()
    pred_confidence = pred[pred_index]
    return pred, pred_index, pred_confidence


def get_imagenet_label(index):
    """Gets the imagenet label of certain index

    Args:
        index (int): the index of the label, between 0 and 999, inclusive.
    """
    label = IMAGENET_LABELS[str(index)]
    return label