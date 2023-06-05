import numpy as np
import torch
import torch.nn.functional as F
from robustness import datasets, model_utils
from safety.utils import utils
from torchvision import models
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def test_untargeted_attack(untargeted_adv_attack, eps=0.01):
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
    adv_image = untargeted_adv_attack(
        image.unsqueeze(0), 
        true_index, 
        model, 
        utils.IMAGENET_NORMALIZE, 
        eps=eps
    ).squeeze(0)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image.squeeze(0))
    adv_label = utils.get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    utils.display_adv_images(
        utils.IMAGENET_DENORMALIZE(image), 
        adv_image,
        (label, confidence),
        (adv_label, adv_confidence),
        channels_first=True,
        denormalize=False
    )


def test_targeted_attack(targeted_adv_attack, target_idx=10, eps=0.01):
    # Load the models
    model = models.resnet18(pretrained=True, progress=False).eval()
    print('')
    
    # Load the preprocessed image
    image, _ = utils.load_example_image(preprocess=True)
    
    # Generate predictions
    _, index, confidence = utils.make_single_prediction(model, image)
    label = utils.get_imagenet_label(index)
    label = label.split(',')[0]

    # Get the target label
    target_label = utils.get_imagenet_label(target_idx)
    target_label = target_label.split(',')[0]
    print(f'The target index corresponds to a label of {target_label}!')

    # Generate Adversarial Example
    target_idx = torch.Tensor([target_idx]).type(torch.long)
    adv_image = targeted_adv_attack(
        image.unsqueeze(0), 
        target_idx, 
        model, 
        utils.IMAGENET_NORMALIZE, 
        eps=eps
    ).squeeze(0)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image.squeeze(0))
    adv_label = utils.get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    utils.display_adv_images(
        utils.IMAGENET_DENORMALIZE(image),
        adv_image,
        (label, confidence),
        (adv_label, adv_confidence),
        channels_first=True,
        denormalize=False
    )


def attack_normal_model(
    targeted_adv_attack, 
    target_idx=10, 
    eps=0.03, 
    num_steps=10, 
    step_size=0.01
):
    # Load the models
    model = models.resnet18(pretrained=True, progress=False).eval()
    print('')
    
    # Load the preprocessed image
    image, _ = utils.load_example_image(preprocess=True)

    # Generate predictions
    _, index, confidence = utils.make_single_prediction(model, image)
    label = utils.get_imagenet_label(index)
    label = label.split(',')[0]

    # Get the target label
    target_label = utils.get_imagenet_label(target_idx)
    target_label = target_label.split(',')[0]
    print(f'The target index corresponds to a label of {target_label}!')

    # Generate Adversarial Example
    target_idx = torch.Tensor([target_idx]).type(torch.long)
    adv_image = targeted_adv_attack(
        image.unsqueeze(0), 
        target_idx, 
        model, 
        utils.IMAGENET_NORMALIZE,
        eps=eps, 
        num_steps=num_steps,
        step_size=step_size
    ).squeeze(0)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image.squeeze(0))
    adv_label = utils.get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    utils.display_adv_images(
        utils.IMAGENET_DENORMALIZE(image),
        adv_image,
        (label, confidence),
        (adv_label, adv_confidence),
        channels_first=True,
        denormalize=False
    )


def get_adv_trained_model():
    attack_model, _ = model_utils.make_and_restore_model(
        arch='resnet18', 
        dataset=datasets.ImageNet(''), 
        resume_path='safety/lesson1/checkpoints/resnet18_linf_eps8.0.ckpt'
    )
    model = attack_model.model
    return model


def attack_adversarially_trained_model(
    targeted_adv_attack, 
    target_idx=10, 
    eps=0.03, 
    num_steps=10, 
    step_size=0.01
):
    # Load the models
    model = get_adv_trained_model().eval()
    print('')
    
    # Load the preprocessed image
    image, _ = utils.load_example_image(preprocess=False)
    image = image / 255

    # Generate predictions
    _, index, confidence = utils.make_single_prediction(model, image)
    label = utils.get_imagenet_label(index)
    label = label.split(',')[0]

    # Get the target label
    target_label = utils.get_imagenet_label(target_idx)
    target_label = target_label.split(',')[0]
    print(f'The target index corresponds to a label of {target_label}!')

    # Generate Adversarial Example
    target_idx = torch.Tensor([target_idx]).type(torch.long)
    adv_image = targeted_adv_attack(
        image.unsqueeze(0), 
        target_idx, 
        model, 
        utils.IMAGENET_NORMALIZE,
        eps=eps, 
        num_steps=num_steps,
        step_size=step_size
    ).squeeze(0)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image.squeeze(0))
    adv_label = utils.get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    utils.display_adv_images(
        utils.IMAGENET_DENORMALIZE(image),
        adv_image,
        (label, confidence),
        (adv_label, adv_confidence),
        channels_first=True,
        denormalize=False
    )

