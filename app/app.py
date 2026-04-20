import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.cm as cm

# Import local architecture functions
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gradcam import make_gradcam_heatmap

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="SkinAI Diagnose Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Polish
st.markdown("""
<style>
    .reportview-container { background-color: #f7f9fa; }
    .stButton>button { width: 100%; border-radius: 6px; padding: 10px; font-weight: bold;}
    .metric-container { font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MAPPINGS & CONFIGURATIONS
# ==========================================
CLASS_MAPPING = {
    0: "Actinic Keratoses (akiec)",
    1: "Basal Cell Carcinoma (bcc)",
    2: "Benign Keratosis-like Lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel) ⚠️",
    5: "Melanocytic Nevi (nv)",
    6: "Vascular Lesions (vasc)"
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.h5")

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
@st.cache_resource
def load_disease_model(path):
    """Loads compiled model once and caches it for rapid inference."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        return None

def process_image(image):
    """Format image exactly as MobileNetV2 expects."""
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized)
    img_array = img_array / 255.0  # Normalize 0-1
    img_batch = np.expand_dims(img_array, axis=0) # Batch of 1
    return img_batch

def overlay_gradcam(original_img, heatmap, alpha=0.5):
    """Takes original PIL image and AI heatmap to construct a visual explanation overlay."""
    # Rescale heatmap to a range 0-255
    heatmap_rescaled = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_rescaled]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + original_img
    
    # Convert numpy backward to PIL image to display easily in Streamlit
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    return superimposed_img

# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
def main():
    
    # --- HEADER ---
    st.markdown("<h1 style='text-align: center;'>🔬 AI-Based Skin Disease Diagnostics System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Powered by MobileNetV2 and Grad-CAM Explainable AI</p>", unsafe_allow_html=True)
    st.divider()

    # --- SIDEBAR & NAVIGATION ---
    with st.sidebar:
        st.header("Control Panel")
        st.info("System Engine: TensorFlow/Keras\n\nDataset: HAM10000")
        
        model = load_disease_model(MODEL_PATH)
        if model:
            st.success("✅ Neural Network Active")
        else:
            st.error("❌ Model Offline. Please train and save 'model.h5' inside 'models/' directory.")
            
    # If model is broken, stop UI here gracefully.
    if model is None: st.stop()

    # --- TAB NAVIGATION ---
    tab1, tab2 = st.tabs(["🩺 Patient Diagnosis (Inference)", "📊 Neural Network Analytics"])

    # ------------------------------------------
    # TAB 1: DIAGNOSIS (INFERENCE & XAI)
    # ------------------------------------------
    with tab1:
        colA, colB = st.columns([1, 1.5], gap="large")
        
        # COLUMN A: Image Selection 
        with colA:
            st.subheader("1. Patient Media Upload")
            st.markdown("Select a dermoscopic standard image for the AI to analyze.")
            uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

            if uploaded_file is not None:
                try:
                    # Original image preserved for Grad-CAM scaling overlay securely
                    img_pil = Image.open(uploaded_file).convert('RGB')
                    st.image(img_pil, caption="Patient Snapshot", use_column_width=True)
                except Exception as e:
                    st.error("Corrupted media file detected.")
                    st.stop()
                    
                analyze_btn = st.button("Initialize Deep Learning Pipeline", type="primary")

        # COLUMN B: AI Analysis Output & Explainability
        with colB:
            if uploaded_file is not None and analyze_btn:
                st.subheader("2. AI Diagnosis Results")
                
                with st.spinner("Processing tensors through MobileNetV2 architecture..."):
                    try:
                        # Preprocess and Predict
                        img_tensor = process_image(img_pil)
                        preds = model.predict(img_tensor)[0] # Extract the 1D prediction array
                        
                        # Identify Top 2 Predictions dynamically
                        top_indices = preds.argsort()[-2:][::-1] # Ascending sort, take last 2, reverse
                        class_1 = CLASS_MAPPING.get(top_indices[0])
                        conf_1 = preds[top_indices[0]] * 100
                        class_2 = CLASS_MAPPING.get(top_indices[1])
                        conf_2 = preds[top_indices[1]] * 100

                        # -- RESULT METRICS --
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.metric("Primary Diagnosis", class_1, f"{conf_1:.2f}% Confidence")
                        with m_col2:
                            st.metric("Secondary Suspect", class_2, f"{conf_2:.2f}% Confidence")

                        # -- EXPLAINABLE AI (GRAD-CAM) --
                        st.divider()
                        st.subheader("3. Explainable AI (Grad-CAM)")
                        st.markdown("The heatmap indicates which pathological features the CNN focused heavily on to establish the primary diagnosis (Red = High focus).")
                        
                        # Generate Heatmap tensor
                        heatmap = make_gradcam_heatmap(img_tensor, model, pred_index=top_indices[0])
                        
                        # Convert PIL img back to format needed for openCV/numpy blending
                        img_arr_raw = keras.utils.img_to_array(img_pil)
                        
                        # Overlay
                        cam_image = overlay_gradcam(img_arr_raw, heatmap, alpha=0.6)
                        
                        st.image(cam_image, caption="AI Attention Heatmap Overlay", use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Inference Engine failed: {e}")
                        
        # Global Disclaimer below interface components inside tab
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning("**LEGAL DISCLAIMER**: This technology is developed as a Major Computer Science / Engineering project. It is strictly a decision-support tool and must not be utilized for self-diagnosis or bypassing trained dermatology medical experts.")

    # ------------------------------------------
    # TAB 2: MODEL PERFORMANCE DASHBOARD
    # ------------------------------------------
    with tab2:
        st.subheader("Deep Learning Model Performance Analytics")
        st.markdown("These empirical charts showcase the architecture's learning curve and capability to generalize over the validation/test sets during its training phase.")
        
        g_col1, g_col2 = st.columns(2)
        
        # Assuming training_history.png and confusion_matrix.png were generated via evaluate.py/train.py
        p1 = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphs", "training_history.png")
        p2 = os.path.join(os.path.dirname(__file__), "..", "outputs", "confusion_matrix.png")
        p3 = os.path.join(os.path.dirname(__file__), "..", "outputs", "classification_report.txt")
        
        with g_col1:
            if os.path.exists(p1):
                st.image(str(p1), caption="Accuracy and Loss Learning Curves")
            else:
                st.info("Training curve visualizations pending generation.")
                
            if os.path.exists(p3):
                with st.expander("Show Raw Classification Report (F1, Precision)"):
                    with open(p3, "r") as f:
                        st.code(f.read())
        
        with g_col2:
            if os.path.exists(p2):
                st.image(str(p2), caption="Unseen Data Testing - Confusion Matrix")
            else:
                st.info("Confusion Matrix visualization pending generation. Run `src/evaluate.py` to populate.")

if __name__ == "__main__":
    main()
