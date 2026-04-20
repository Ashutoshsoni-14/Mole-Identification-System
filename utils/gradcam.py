import numpy as np
import tensorflow as tf
import keras

def get_last_conv_layer_name(model):
    """
    Dynamically finds the name of the last convolutional layer in a Keras model.
    This works by searching backwards from the output for a layer that outputs a 4D tensor.
    """
    for layer in reversed(model.layers):
        if hasattr(layer, 'output'):
            # Convolutional feature maps are 4D (batch, height, width, channels)
            if len(layer.output.shape) == 4:
                return layer.name
    raise ValueError("Could not find a valid Convolutional layer for Grad-CAM. Check model architecture.")

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Generates a Grad-CAM heatmap highlighting the regions of the image that 
    the AI model focused on to make its specific prediction.
    """
    last_conv_layer_name = get_last_conv_layer_name(model)
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        inputs=[model.inputs], 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # and discard negative values (we only care about features that positively impact the class)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
