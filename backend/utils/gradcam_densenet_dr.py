import numpy as np
import tensorflow as tf
import cv2

def generate_gradcam_densenet(model, image_tensor, class_idx, layer_name=None):
    """
    Generate Grad-CAM heatmap for a Keras DenseNet model.
    
    Args:
        model: Keras model
        image_tensor: Preprocessed image tensor of shape [1, 224, 224, 3]
        class_idx: Index of predicted class
        layer_name: Name of the convolutional layer to use for Grad-CAM

    Returns:
        A 2D normalized Grad-CAM heatmap (shape: [224, 224])
    """
    if layer_name is None:
        # Pick the last convolutional layer in DenseNet
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        loss = predictions[:, class_idx]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients over the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply channel-wise weights with conv output
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))

    return heatmap
