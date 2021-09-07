# Everything Necessary for the Adversarial Examples Class
import matplotlib.pyplot as plt
import numpy as np
import torch
from safety.utils import utils
from torchvision import models


def test_untargeted_FGSM(untargeted_FGSM):
    # Load the models
    model = models.resnet18(pretrained=True, progress=False).eval()
    print('')
    
    # Load the preprocessed image
    image, true_index = utils.load_example_image(preprocess=True)

    # Generate predictions
    _, index, confidence = utils.make_single_prediction(model, image)
    label = utils.get_imagenet_label(index)
    label = label.split(',')[0]

    # Generate Adversarial Example
    true_index = torch.Tensor([true_index]).type(torch.long)
    adv_image = untargeted_FGSM(image, true_index, model, 0.01)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image)
    adv_label = utils.get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    display_image = utils.IMAGENET_DENORMALIZE(image).detach().cpu().numpy()
    display_image = np.moveaxis(display_image, 0, 2)
    display_adv_image = utils.IMAGENET_DENORMALIZE(adv_image).detach().cpu().numpy()
    display_adv_image = np.moveaxis(display_adv_image, 0, 2)
    _, axes = plt.subplots(1,2)
    axes[0].imshow(display_image)
    axes[0].set_title('Original Image')
    axes[0].set(xlabel=f'Prediction: {label}\nConfidence: {confidence:.4f}')
    axes[1].imshow(display_adv_image)
    axes[1].set_title('Adversarial Image')
    axes[1].set(xlabel=f'Prediction: {adv_label}\nConfidence: {adv_confidence:.4f}')
    plt.show()