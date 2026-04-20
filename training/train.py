import os
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import get_data_generators
from utils.model import build_model

def plot_training_history(history, output_path):
    """
    Plots the training vs validation accuracy and loss curves.
    Saves the graphic to disk.
    """
    print("\n--- Generating Training Artifacts ---")
    
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='green', marker='x')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='red', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange', marker='x')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Save to disk
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training history saved accurately at: {output_path}")

def main():
    # ---------------------------------------------
    # PATHS AND CONFIGURATIONS
    # ---------------------------------------------
    WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset") # Was data/structured_dataset, now dataset directly
    MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
    OUTPUTS_DIR = os.path.join(WORKSPACE_DIR, "outputs")
    GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "model.h5")
    HISTORY_PLOT_PATH = os.path.join(GRAPHS_DIR, "training_history.png")
    
    EPOCHS = 15

    # 1. Fetch Generators
    train_gen, val_gen, test_gen = get_data_generators(DATASET_DIR)
    
    if train_gen is None:
        print("Failed to initialize data. Exiting...")
        return

    # Extract dynamic properties
    num_classes = len(train_gen.class_indices)
    input_shape = train_gen.image_shape

    # 2. Build and Compile Model
    model = build_model(input_shape=input_shape, num_classes=num_classes)

    # 3. Setup Callbacks
    # Early stopping prevents overtraining if validation loss stops improving
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5,             # Wait 5 epochs before stopping
        restore_best_weights=True,
        verbose=1
    )
    
    # Checkpoint ensures we always save the model state with the lowest val_loss
    model_checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # 4. Start Training Process
    print(f"\n--- Starting Model Training for {EPOCHS} Epochs ---")
    
    # Calculate step intervals safely (crucial if mocking small datasets)
    steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
    validation_steps = max(1, val_gen.samples // val_gen.batch_size)

    try:
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=[early_stop, model_checkpoint]
        )
        
        # 5. Output Graphics
        plot_training_history(history, HISTORY_PLOT_PATH)
        print("\nSUCCESS: Model successfully completed training sequence.")
        
    except Exception as e:
        print(f"\nTraining Failed: {e}")

if __name__ == "__main__":
    main()
