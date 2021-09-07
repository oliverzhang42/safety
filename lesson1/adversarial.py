# Everything Necessary for the Adversarial Examples Class
import torch
import matplotlib.pyplot as plt
from safety.utils import utils
from torchvision import models


def test_untargeted_FGSM(untargeted_FGSM):
    # Load the models
    model = models.resnet18(pretrained=True, progress=False).eval()
    print('')
    
    # Load the preprocessed image
    image, true_index = utils.load_example_image(preprocess=True)
    true_label = utils.get_imagenet_label(true_index)
    true_label = true_label.split(',')[0]
    print(f'The correct label is "{true_label}" \n')

    # Generate predictions
    _, index, confidence = utils.make_single_prediction(model, image)
    label = utils.get_imagenet_label(index)
    label = label.split(',')[0]
    print(f'The original prediction was {label} with logit value {confidence}. \n')

    # Generate Adversarial Example
    true_index = torch.Tensor([true_index]).type(torch.long)
    adv_image = untargeted_FGSM(image, true_index, model, 0.01)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image)
    adv_label = utils.get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]
    print(f'The prediction on the adversarial image was {adv_label} with logit value {adv_confidence}.')

    # Display Images
    display_image = utils.IMAGENET_DENORMALIZE(image).detach().cpu().numpy()
    display_adv_image = utils.IMAGENET_DENORMALIZE(adv_image).detach().cpu().numpy()
    _, axes = plt.subplots(1,2)
    axes[0,0].imshow(display_image)
    axes[0,1].imshow(display_adv_image)
    plt.show()