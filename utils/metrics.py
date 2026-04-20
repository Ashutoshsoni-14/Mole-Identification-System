import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn.metrics import confusion_matrix, classification_report
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import get_data_generators

def perform_evaluation(model_path, dataset_dir, output_dir):
    """
    Evaluates the trained model against the unseen test dataset.
    Generates a Confusion Matrix and a Classification Report (Precision, Recall, F1).
    """
    print("\n--- Starting Formal Model Evaluation ---")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return

    # 1. Load Model
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    print("Model loaded successfully.")

    # 2. Get Test Data (No Augmentations)
    _, _, test_gen = get_data_generators(dataset_dir)
    
    if test_gen is None:
        print("Test data could not be loaded.")
        return

    # Extract class mapping
    class_labels = list(test_gen.class_indices.keys())
    
    # Reset generator to ensure alignment
    test_gen.reset()

    # 3. Generate Predictions Make sure steps cover exactly the whole dataset
    steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
    if steps == 0: steps = 1 # Fallback for dummy data testing
        
    print("Generating predictions on Test Set. This may take a moment...")
    predictions = model.predict(test_gen, steps=steps, verbose=1)
    
    # Get highest confidence prediction index
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels directly from the generator
    # For small dummy datasets this works safely, for large batches we rely on classes array
    y_true = test_gen.classes

    # Handle mismatched sizes if batches wrap unevenly
    if len(y_pred) > len(y_true):
        y_pred = y_pred[:len(y_true)]

    # 4. Generate Classification Report (Precision, Recall, F1-Score)
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=0)
    print(report)
    
    # Save text report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write("Evaluation Classification Report - AI Skin Disease Diagnosis\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        print(f"Saved classification report to {output_dir}/classification_report.txt")

    # 5. Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title('Confusion Matrix - Skin Disease Classification', fontsize=16)
    plt.ylabel('Actual Medical Diagnosis', fontsize=12)
    plt.xlabel('AI Predicted Diagnosis', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"Saved Confusion Matrix visualization to {cm_path}")
    print("\nEvaluation Phase Complete.")

if __name__ == "__main__":
    WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset")
    MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
    OUTPUTS_DIR = os.path.join(WORKSPACE_DIR, "outputs")
    
    MODEL_PATH = os.path.join(MODELS_DIR, "model.h5")
    
    perform_evaluation(MODEL_PATH, DATASET_DIR, OUTPUTS_DIR)
