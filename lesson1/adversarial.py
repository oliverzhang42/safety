# Everything Necessary for the Adversarial Examples Class
import json
import numpy as np
import torch
import torchvision

def get_imagenet_labels():
    with open('safety/lesson1/imagenet_labels.json', 'r') as f:
        index_to_label = json.load(f)
    return index_to_label

def test_FGSM(FGSM_Implementation):
    pass