import tensorflow as tf
import keras
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model

def build_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Builds a Transfer Learning model using MobileNetV2.
    
    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): Number of target classes to predict.
        
    Returns:
        compiled Keras model ready for training.
    """
    print("\n--- Constructing AI Model Architecture ---")
    
    # 1. Load Pretrained MobileNetV2 without the top classification layers
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 2. Freeze the base model to retain learned features (no weight updates for these layers initially)
    base_model.trainable = False
    print("Pretrained MobileNetV2 initialized and base layers visually frozen.")
    
    # 3. Add Custom Classification Head
    # Extract features from the base model
    x = base_model.output
    
    # GlobalAveragePooling reduces the spatial dimensions (flattens 2D feature maps to 1D)
    x = GlobalAveragePooling2D(name='global_average_pooling')(x)
    
    # Add a fully connected layer to interpret features
    x = Dense(512, activation='relu', name='dense_projection')(x)
    
    # Add dropout for regularization to prevent overfitting on the custom head
    x = Dropout(0.5, name='dropout_regularization')(x)
    
    # Final Output Layer with Softmax for multi-class predictions
    outputs = Dense(num_classes, activation='softmax', name='classifier_output')(x)
    
    # 4. Construct Final Model Mapping Inputs -> Outputs
    model = Model(inputs=base_model.input, outputs=outputs, name='Skin_Disease_Classifier')
    
    # 5. Compile Model
    # Using Adam optimizer, Categorical Crossentropy for one-hot labels, tracking Accuracy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Compiled Successfully!")
    print(f"Total Parameters: {model.count_params():,}")
    
    return model

if __name__ == "__main__":
    # Test compilation to catch syntax/layer logic errors locally
    mock_model = build_model()
    mock_model.summary()
