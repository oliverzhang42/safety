# Everything Necessary for the Adversarial Examples Class
import matplotlib.pyplot as plt
import numpy as np
import torch
from safety.utils import utils
from torchvision import models


def test_untargeted_FGSM(untargeted_FGSM, eps=0.01):
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
    adv_image = untargeted_FGSM(image.unsqueeze(0), true_index, model, eps).squeeze(0)

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


def test_targeted_FGSM(targeted_FGSM, target_idx=10, eps=0.01):
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
    adv_image = targeted_FGSM(image.unsqueeze(0), [target_idx], model, eps).squeeze(0)

    # Display Results
    _, adv_index, adv_confidence = utils.make_single_prediction(model, adv_image.unsqueeze(0))
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


def untargeted_PGD():
    pass

def targeted_PGD():
    pass

'''
class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx


class PGDTarget(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        random_by = np.random.randint(0, 1000, by.size(0))
        for i in range(len(random_by)):
            while random_by[i] == by[i]:
                random_by[i] = np.random.randint(1000)
        random_by = torch.LongTensor(random_by).cuda()

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx * 2 - 1)
                loss = -F.cross_entropy(logits, random_by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx
'''