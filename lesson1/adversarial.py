# Everything Necessary for the Adversarial Examples Class
import json
import numpy as np
import torch
import torchvision
from safety.utils import utils
from torch import nn
from torchvision import models


def test_untargeted_FGSM(untargeted_FGSM):
  # Load the models
  model = models.resnet18(pretrained=True, progress=False).eval()
  print('')
  
  # Load the preprocessed image
  image, true_index = utils.load_example_image(preprocess=True)
  true_label = utils.get_imagenet_label(true_index)
  print(f'The correct label is "{true_label}" \n')

  # Generate predictions
  _, index, confidence = utils.make_single_prediction(model, image)
  label = utils.get_imagenet_label(index)
  print(f'The original prediction was {label} with logit value {confidence}. \n')

  # Generate Adversarial Example
  true_index = torch.Tensor([true_index]).type(torch.long)
  adv_image = untargeted_FGSM(image, true_index, model, 0.01)

  # Display Results
  _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image)
  adv_label = utils.get_imagenet_label(adv_index)
  print(f'The adversarial prediction was {adv_label} with logit value {adv_confidence}.')
