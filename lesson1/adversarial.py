# Everything Necessary for the Adversarial Examples Class
import matplotlib.pyplot as plt
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

    # Scaling up the epsilon to adjust for normalization
    eps = eps * 1/0.22

    # Generate Adversarial Example
    true_index = torch.Tensor([true_index]).type(torch.long)
    adv_image = untargeted_adv_attack(image.unsqueeze(0), true_index, model, eps=eps).squeeze(0)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image.squeeze(0))
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


def test_targeted_attack(targeted_adv_attack, target_idx=10, eps=0.01):
    # Load the models
    model = models.resnet18(pretrained=True, progress=False).eval()
    print('')
    
    # Load the preprocessed image
    image, _ = utils.load_example_image(preprocess=True)
    
    # Scaling up the epsilon to adjust for normalization
    eps = eps * 1/0.22
    
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
    adv_image = targeted_adv_attack(image.unsqueeze(0), target_idx, model, eps=eps).squeeze(0)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image.squeeze(0))
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


def untargeted_PGD(x_batch, true_labels, network, num_steps=10, step_size=0.003, eps=0.01):
    """Generates a batch of untargeted PGD adversarial examples

    Args:
        x_batch (torch.Tensor): the batch of preprocessed input examples.
        true_labels (torch.Tensor): the batch of true labels of the example.
        network (nn.Module): the network to attack.
        num_steps (int): the number of steps to run PGD.
        step_size (float): the size of each PGD step.
        eps (float): the bound on the perturbations.
    """
    adv_x = x_batch.detach()
    adv_x += torch.zeros_like(adv_x).uniform_(eps, eps)

    for i in range(num_steps):
        adv_x.requires_grad_()
        
        # Calculate gradients
        with torch.enable_grad():
            logits = network(adv_x) #TODO: Removed the 2*x - 1
            loss = F.cross_entropy(logits, true_labels, reduction='sum')
        grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]
        
        # Perform one gradient step
        adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())
        
        # Project the image to the ball
        adv_x = torch.maximum(adv_x, x_batch - eps)
        adv_x = torch.minimum(adv_x, x_batch + eps)

    return adv_x


def targeted_PGD(x_batch, target_labels, network, num_steps=10, step_size=0.01, eps=0.03):
    """Generates a batch of untargeted PGD adversarial examples

    Args:
        x_batch (torch.Tensor): the batch of preprocessed input examples.
        target_labels (torch.Tensor): the labels the model will predict after the attack.
        network (nn.Module): the network to attack.
        num_steps (int): the number of steps to run PGD.
        step_size (float): the size of each PGD step.
        eps (float): the bound on the perturbations.
    """
    adv_x = x_batch.detach()
    adv_x += torch.zeros_like(adv_x).uniform_(eps, eps)

    for i in range(num_steps):
        adv_x.requires_grad_()
        
        # Calculate gradients
        with torch.enable_grad():
            logits = network(adv_x)
            loss = F.cross_entropy(logits, target_labels, reduction='sum')
        grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]
        
        # Perform one gradient step
        # Note that this time we use gradient descent instead of gradient ascent
        adv_x = adv_x.detach() - step_size * torch.sign(grad.detach())
        
        # Project the image to the ball
        adv_x = torch.maximum(adv_x, x_batch - eps)
        adv_x = torch.minimum(adv_x, x_batch + eps)

    return adv_x


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

    # Scaling up the epsilon to adjust for normalization
    eps = eps * 1/0.22

    # Generate Adversarial Example
    target_idx = torch.Tensor([target_idx]).type(torch.long)
    adv_image = targeted_adv_attack(
        image.unsqueeze(0), 
        target_idx, 
        model, 
        eps=eps, 
        num_steps=num_steps,
        step_size=step_size
    ).squeeze(0)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image.squeeze(0))
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


def attack_adversarially_trained_model(
    targeted_adv_attack, 
    target_idx=10, 
    eps=0.03, 
    num_steps=10, 
    step_size=0.01
):
    # Load the models
    attack_model, _ = model_utils.make_and_restore_model(
        arch='resnet18', 
        dataset=datasets.ImageNet(''), 
        resume_path='safety/lesson1/resnet18_linf_eps8.0.ckpt'
    )
    model = attack_model.model
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
        eps=eps, 
        num_steps=num_steps,
        step_size=step_size
    ).squeeze(0)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image.squeeze(0))
    adv_label = utils.get_imagenet_label(adv_index)
    adv_label = adv_label.split(',')[0]

    # Display Images
    display_image = image.detach().cpu().numpy()
    display_image = np.moveaxis(display_image, 0, 2)
    display_adv_image = adv_image.detach().cpu().numpy()
    display_adv_image = np.moveaxis(display_adv_image, 0, 2)
    _, axes = plt.subplots(1,2)
    axes[0].imshow(display_image)
    axes[0].set_title('Original Image')
    axes[0].set(xlabel=f'Prediction: {label}\nConfidence: {confidence:.4f}')
    axes[1].imshow(display_adv_image)
    axes[1].set_title('Adversarial Image')
    axes[1].set(xlabel=f'Prediction: {adv_label}\nConfidence: {adv_confidence:.4f}')
    plt.show()

