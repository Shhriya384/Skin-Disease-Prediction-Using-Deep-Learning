import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import webbrowser

# -------------------------------
# 1️⃣ Load Trained Model (FIXED)
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "skin_disease_model.h5")  # ✅ correct model name
model = load_model(model_path)

# -------------------------------
# 2️⃣ Class Names (MUST match training order)
# -------------------------------
class_names = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis Photos',
    'Bullous Disease Photos',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos',
    'Exanthems and Drug Eruptions',
    'Hair Loss Photos Alopecia and other Hair Diseases',
    'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles',
    'Nail Fungus and other Nail Disease',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Scabies Lyme Disease and other Infestations and Bites',
    'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Vascular Tumors',
    'Vasculitis Photos',
    'Warts Molluscum and other Viral Infections'
]

# -------------------------------
# 3️⃣ Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Skin Disease Detection", layout="centered")

st.title("🩺 AI-Based Skin Disease Detection System")
st.write("Upload a skin image to analyze the possible disease category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # 4️⃣ Preprocess Image
    # -------------------------------
    img_resized = img.resize((224, 224))
    x = img_to_array(img_resized)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # -------------------------------
    # 5️⃣ Prediction
    # -------------------------------
    preds = model.predict(x)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    confidence = round(confidence, 2)

    predicted_class = class_names[class_idx]

    # -------------------------------
    # 6️⃣ Dynamic Severity (Better Logic)
    # -------------------------------
    if confidence > 85:
        severity = "High"
    elif confidence > 60:
        severity = "Moderate"
    else:
        severity = "Low"

    # -------------------------------
    # 7️⃣ Display Results
    # -------------------------------
    st.success(f"### 🧾 Predicted Disease: {predicted_class}")
    st.info(f"### 📊 Confidence Score: {confidence}%")

    if severity == "High":
        st.error(f"### 🔴 Severity Level: {severity}")
    elif severity == "Moderate":
        st.warning(f"### 🟡 Severity Level: {severity}")
    else:
        st.success(f"### 🟢 Severity Level: {severity}")

    # -------------------------------
    # 8️⃣ Advisory Message
    # -------------------------------
    st.write("### 💡 Suggested Action:")

    if severity == "High":
        st.write("⚠️ Immediate dermatologist consultation recommended.")
    elif severity == "Moderate":
        st.write("It is advised to consult a dermatologist for proper evaluation.")
    else:
        st.write("Maintain hygiene and monitor symptoms. Consult a doctor if it persists.")

    

    # -------------------------------
    # 🔎 Find Dermatologist Button
    # -------------------------------
    if st.button("Find Nearby Dermatologist"):
        webbrowser.open("https://www.google.com/search?q=dermatologist+near+me")