import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.cm as cm
import datetime
import base64

# Import local architecture functions
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gradcam import make_gradcam_heatmap

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SkinAI Diagnosix Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PREMIUM EXPERIMENTAL CSS STYLING
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;800&display=swap');
    
    /* Global Styling & Animations */
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    @keyframes slideDownFade {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Premium Header with animated gradient */
    .premium-header {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #000000);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite, slideDownFade 0.8s ease-out forwards;
        padding: 3rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        margin-bottom: 2.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .premium-header h1 {
        margin: 0;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: -webkit-linear-gradient(45deg, #ffffff, #a1c4fd, #c2e9fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .premium-header p {
        margin-top: 15px;
        font-size: 1.2rem;
        font-weight: 300;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #89b4c4;
    }

    /* Subtle hover animations on cards */
    div[data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"] {
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
    }
    div[data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.06);
    }

    /* Metric Values Customization overriden to pop */
    div[data-testid="stMetricValue"] {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #ff0844 0%, #ffb199 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Styled Buttons with pulse and float */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        padding: 0.8rem;
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stButton>button:hover {
        transform: scale(1.02) translateY(-3px);
        box-shadow: 0 12px 25px rgba(118, 75, 162, 0.4);
    }
    
    /* Section Headers with expanding animated underlines */
    .section-title {
        color: #1e293b;
        font-weight: 800;
        font-size: 1.6rem;
        position: relative;
        padding-bottom: 12px;
        margin-bottom: 25px;
        display: inline-block;
        animation: slideDownFade 0.6s ease-out forwards;
    }
    .section-title::after {
        content: '';
        position: absolute;
        left: 0;
        bottom: 0;
        width: 40px;
        height: 5px;
        background: linear-gradient(90deg, #ff0844, #ffb199);
        border-radius: 5px;
        transition: width 0.4s ease-in-out;
    }
    .section-title:hover::after {
        width: 100%;
    }
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

# Initialize Session State for Prediction History
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
@st.cache_resource
def load_disease_model(path):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        return None

def process_image(image):
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized)
    img_array = img_array / 255.0  
    img_batch = np.expand_dims(img_array, axis=0) 
    return img_batch

def overlay_gradcam(original_img, heatmap, alpha=0.5):
    heatmap_rescaled = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_rescaled]
    
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * alpha + original_img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    return superimposed_img

# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
def main():
    
    # --- PREMIUM HEADER ---
    st.markdown("""
    <div class="premium-header">
        <h1>🧬 SkinAI Diagnosix Pro</h1>
        <p>A Deep Learning Architecture for Precision Dermoscopic Classification</p>
    </div>
    """, unsafe_allow_html=True)

    # --- ENHANCED SIDEBAR ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🏥 Project Info</h2>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        * **Engine:** TensorFlow 2.x & Keras
        * **Network:** Custom MobileNetV2
        * **Explainability:** Grad-CAM
        * **Interface:** Streamlit Premium
        """)
        
        st.markdown("### 📋 Instructions")
        st.info("1. Navigate to the 'Diagnostics' tab.\n2. Upload a top-down clear dermoscopic image.\n3. Click Analyze to trigger the CNN.")

        st.markdown("### 👨‍💻 Developed By")
        st.success("**Final Year Project Team**\n\nAI Architecture & UI Design Showcase")
        
        st.divider()
        model = load_disease_model(MODEL_PATH)
        if model:
            st.success("🟢 CNN Core Active")
        else:
            st.error("🔴 Fatal: `model.h5` core offline.")
            
    if model is None: st.stop()

    # --- TAB NAVIGATION ---
    tab1, tab2, tab3 = st.tabs(["🩺 Diagnostics Engine", "📚 Patient History", "📊 Architect Analytics"])

    # ------------------------------------------
    # TAB 1: DIAGNOSTICS & GRAD-CAM
    # ------------------------------------------
    with tab1:
        st.markdown("<h3 class='section-title'>Clinical Image Analysis Workspace</h3>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1.5], gap="large")
        
        # COLUMN A: Upload & Display
        with colA:
            with st.container(border=True):
                st.markdown("#### 1️⃣ Import Lesion Imagery")
                uploaded_file = st.file_uploader("High resolution JPG/PNG optimal.", type=['jpg', 'jpeg', 'png'])

                if uploaded_file:
                    try:
                        img_pil = Image.open(uploaded_file).convert('RGB')
                        st.image(img_pil, caption="Uploaded Dematoscopic Geometry", use_container_width=True)
                        analyze_btn = st.button("🚀 Analyze Clinical Image", type="primary")
                    except Exception:
                        st.error("Corrupted media file detected.")
                        st.stop()
                else:
                    st.info("Awaiting medical imagery for classification.")
                    analyze_btn = False

        # COLUMN B: Engine Results & XAI
        with colB:
            if uploaded_file and analyze_btn:
                
                with st.spinner("⏳ Compiling layer activations and computing gradients..."):
                    try:
                        img_tensor = process_image(img_pil)
                        preds = model.predict(img_tensor)[0]
                        
                        top_indices = preds.argsort()[-2:][::-1] 
                        class_1 = CLASS_MAPPING.get(top_indices[0])
                        conf_1 = preds[top_indices[0]] * 100
                        class_2 = CLASS_MAPPING.get(top_indices[1])
                        conf_2 = preds[top_indices[1]] * 100

                        # Save to Session State History
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state['history'].append({
                            'time': timestamp,
                            'name': uploaded_file.name,
                            'primary': class_1,
                            'confidence': f"{conf_1:.2f}%"
                        })

                        with st.container(border=True):
                            st.markdown("#### 2️⃣ Neural Network Output")
                            # Premium Metric display
                            m_col1, m_col2 = st.columns(2)
                            with m_col1:
                                st.metric(label="Primary Diagnosis", value=class_1.split('(')[0], delta=f"{conf_1:.2f}% Confidence", delta_color="off")
                            with m_col2:
                                st.metric(label="Secondary Suspect", value=class_2.split('(')[0], delta=f"{conf_2:.2f}% Confidence", delta_color="normal")
                                
                        # EXPLAINABLE AI SECTION
                        with st.container(border=True):
                            st.markdown("#### 3️⃣ Model Explainability (Grad-CAM)")
                            st.caption("*Gradient-weighted Class Activation Mapping computes how severely morphological features impacted the network's local minimum.*")
                            
                            heatmap = make_gradcam_heatmap(img_tensor, model, pred_index=top_indices[0])
                            img_arr_raw = keras.utils.img_to_array(img_pil)
                            cam_image = overlay_gradcam(img_arr_raw, heatmap, alpha=0.6)
                            
                            st.image(cam_image, caption="AI Heatmap: Deep Red denotes critical diagnostic regions.", use_container_width=True)

                        st.toast('Analysis Completed Successfully!', icon='✅')
                    except Exception as e:
                        st.error(f"Inference Engine failed during compilation: {e}")

    # ------------------------------------------
    # TAB 2: SESSION PREDICTION HISTORY
    # ------------------------------------------
    with tab2:
        st.markdown("<h3 class='section-title'>Session Audit Log</h3>", unsafe_allow_html=True)
        if len(st.session_state['history']) == 0:
            st.info("No scans have been processed in the current session.")
        else:
            st.markdown("Recent queries executed against the neural network:")
            for item in reversed(st.session_state['history']):
                st.success(f"**[{item['time']}]** Image `{item['name']}` | **Diagnosis:** {item['primary']} | **Confidence:** {item['confidence']}")
                
    # ------------------------------------------
    # TAB 3: ANALYTICS & METRICS
    # ------------------------------------------
    with tab3:
        st.markdown("<h3 class='section-title'>Network Capability & Performance</h3>", unsafe_allow_html=True)
        st.markdown("Empirical testing metrics accumulated during the validation cycle.")
        
        g_col1, g_col2 = st.columns(2)
        
        p1 = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphs", "training_history.png")
        p2 = os.path.join(os.path.dirname(__file__), "..", "outputs", "confusion_matrix.png")
        
        with g_col1:
            with st.container(border=True):
                st.markdown("#### Learning Curves")
                if os.path.exists(p1): st.image(str(p1), use_container_width=True)
                else: st.warning("Graph not deployed to outputs logic yet.")
        
        with g_col2:
            with st.container(border=True):
                st.markdown("#### Testing Confusion Matrix")
                if os.path.exists(p2): st.image(str(p2), use_container_width=True)
                else: st.warning("Matrix pending deployment generation.")

if __name__ == "__main__":
    main()
