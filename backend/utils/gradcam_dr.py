# utils/gradcam_dr.py

import torch
import numpy as np
from torch.nn import functional as F

def generate_gradcam(model, input_tensor, class_idx):
    """
    Generate Grad-CAM heatmap from a ViT model (torchvision ViT).
    Assumes input_tensor is of shape [1, 3, 224, 224] and model is vit_b_16.
    """

    # Store gradients and features
    gradients = []
    activations = []

    def save_gradient_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation_hook(module, input, output):
        activations.append(output)

    # Register hooks to the last encoder block output
    last_block = model.encoder.layers[-1]
    handle_act = last_block.register_forward_hook(save_activation_hook)
    handle_grad = last_block.register_full_backward_hook(save_gradient_hook)

    # Forward pass
    model.zero_grad()
    output = model(input_tensor)  # [1, num_classes]
    pred_class_score = output[0, class_idx]
    pred_class_score.backward(retain_graph=True)

    # Remove hooks
    handle_act.remove()
    handle_grad.remove()

    # Get stored values
    grad = gradients[0]          # [1, tokens, hidden_dim]
    act = activations[0]         # [1, tokens, hidden_dim]

    # Global average pooling over gradients
    weights = grad.mean(dim=1, keepdim=True)  # [1, 1, hidden_dim]

    # Weighted sum of activations
    cam = (weights * act).sum(dim=2).squeeze(0)  # [tokens]

    # Remove class token
    cam = cam[1:]  # skip class token

    # Reshape to 14x14
    cam = cam.reshape(14, 14).detach().cpu().numpy()

    # Normalize between 0 and 1
    cam = np.maximum(cam, 0)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam
