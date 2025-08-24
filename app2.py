import streamlit as st
import pickle
import pandas as pd
import numpy as np



# --- Premium IntelliCare Celestial v6 Theme ---
st.markdown("""
    <style>
        /* === IntelliCare Celestial v6 === */
        :root {
            --primary-color: #FDB813;  
            --primary-hover: #FFD700;  
            --primary-glow: rgba(253, 184, 19, 0.15);
            --secondary-accent: #4A90E2;
            --bg-start: #0f172a; 
            --bg-mid: #1e293b;  
            --bg-end: #0c1322;
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.1);
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-accent: var(--primary-color);
            --accent-warning: #F39C12;
            --font-family: 'Inter', system-ui, sans-serif;
        }

        body, .stApp {
            font-family: var(--font-family);
            background-image: linear-gradient(160deg, var(--bg-start) 10%, var(--bg-mid) 50%, var(--bg-end) 90%);
            color: var(--text-primary);
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Page config
st.set_page_config(page_title="DiagnoX AI", page_icon="🩺", layout="wide")

# ----------------------------
# Load model and data
# ----------------------------
@st.cache_data
def load_data():
    try:
        with open("disease_predictor.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception:
        st.error("❌ Error loading model. Ensure 'disease_predictor.pkl' exists and is valid.")
        st.stop()

    try:
        medications_df = pd.read_csv("medications.csv")
    except Exception:
        st.error("❌ Error loading 'medications.csv'. Ensure the file exists in the app directory.")
        st.stop()

    try:
        train_df = pd.read_csv("Training.csv")
        if "Unnamed: 133" in train_df.columns:
            train_df = train_df.drop("Unnamed: 133", axis=1)
    except Exception:
        st.error("❌ Error loading 'Training.csv'. Ensure the file exists in the app directory.")
        st.stop()

    symptoms = train_df.drop("prognosis", axis=1).columns.tolist()
    return model, medications_df, symptoms

# Load everything
model, medications_df, symptoms = load_data()

# ----------------------------
# ✅ Inject your full HTML content here
# ----------------------------
st.markdown("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DiagnoX AI | Your Personal Health Symptom Analyzer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link 
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" 
        rel="stylesheet">
    <script src="https://unpkg.com/@phosphor-icons/web"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/choices.js/public/assets/styles/choices.min.css"/>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="app-header">
            <div class="logo-section">
                <i class="ph-bold ph-activity logo"></i>
                <h1 class="brand-title">DiagnoX AI</h1>
            </div>
            <p class="subtitle">
                Predict potential health conditions with the power of AI.  
                Select your symptoms below to begin your analysis.
            </p>
        </header>

        <!-- Placeholder where Streamlit widget renders -->
        <main class="glass-card main-card">
            <div id="streamlit-form-placeholder"></div>
        </main>

        <!-- Info Sections -->
        <section class="content-section">
            <div class="glass-card content-card">
                <i class="ph-bold ph-robot content-icon"></i>
                <h2>How It Works</h2>
                <p>
                    DiagnoX AI leverages a highly accurate <strong>Random Forest model</strong>, trained on thousands of anonymized patient records.  
                    When you select your symptoms, the model analyzes patterns and predicts the most likely health condition.
                </p>
            </div>
            <div class="glass-card content-card">
                <i class="ph-bold ph-gear-six content-icon"></i>
                <h2>Features You’ll Love</h2>
                <ul class="feature-list">
                    <li><i class="ph-fill ph-check-circle"></i> Instant Predictions</li>
                    <li><i class="ph-fill ph-check-circle"></i> High Accuracy</li>
                    <li><i class="ph-fill ph-check-circle"></i> Easy to Use</li>
                    <li><i class="ph-fill ph-check-circle"></i> Actionable Insights</li>
                </ul>
            </div>
        </section>

        <!-- Footer -->
        <footer>
            <div class="footer-content">
                <p>Made with <span class="heart">❤️</span> by <strong>Vansh Malhotra</strong></p>
                <div class="social-links">
                    <a href="https://www.linkedin.com/in/vanshmalhotra1301/" target="_blank">
                        <i class="ph-bold ph-linkedin-logo"></i>
                    </a>
                    <a href="https://github.com/VanshMalhotra1301" target="_blank">
                        <i class="ph-bold ph-github-logo"></i>
                    </a>
                    <a href="#"><i class="ph-bold ph-globe"></i></a>
                </div>
            </div>
            <p class="copyright">&copy; 2025 DiagnoX AI. All Rights Reserved.</p>
        </footer>
    </div>
</body>
</html>
""", unsafe_allow_html=True)

# ----------------------------
# Streamlit Form (injected into placeholder above)
# ----------------------------
st.markdown("<div id='symptom-form-section'>", unsafe_allow_html=True)
st.subheader("⚡ Input Symptoms")
selected_symptoms = st.multiselect(
    "Choose symptoms you are experiencing:",
    options=symptoms,
    help="Start typing to search symptoms.",
)

predict_btn = st.button("🔍 Predict", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Prediction Logic
# ----------------------------
if predict_btn:
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom before predicting.")
    else:
        input_data = [0] * len(symptoms)
        for s in selected_symptoms:
            input_data[symptoms.index(s)] = 1
        input_data = np.array(input_data).reshape(1, -1)

        try:
            prediction = model.predict(input_data)[0]
            st.success(f"✅ Predicted Disease: **{prediction}**")

            suggestion = medications_df[medications_df["Disease"].str.lower() == prediction.lower()]["Suggestion"].tolist()
            if suggestion:
                st.subheader("💊 Suggested Medications / Advice:")
                for s in suggestion:
                    st.write(f"- {s}")
            else:
                st.info("No suggestions found for this disease in medications.csv.")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
else:
    st.info("👆 Select symptoms above and click **Predict**.")
