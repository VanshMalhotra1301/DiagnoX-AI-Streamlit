import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Configuration ---
# Set page config once at the beginning
st.set_page_config(
    page_title="DiagnoX AI | Advanced Symptom Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* === DiagnoX AI - Sovereign Dark Gold v3.0 === */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

:root {
    --primary-gold: #D4AF37;
    --gold-hover: #FFD700;
    --gold-glow: rgba(212, 175, 55, 0.25);

    --bg-dark-1: #0a0a0a;
    --bg-dark-2: #141414;
    --bg-dark-3: #1a1a1a;

    --card-bg: rgba(20, 20, 20, 0.65);
    --card-border: rgba(212, 175, 55, 0.15);

    --text-primary: #f5f5f5;
    --text-secondary: #999;
    --font-family: 'Poppins', sans-serif;
}

/* === Animated Cosmic Gradient Background === */
@keyframes auroraShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp {
    font-family: var(--font-family);
    background: linear-gradient(135deg, #0a0a0a, #141414, #1a1a1a, #0f0f0f);
    background-size: 400% 400%;
    animation: auroraShift 40s ease infinite;
    color: var(--text-primary);
    position: relative;
    overflow: hidden;
}

/* === Floating Particles Animation === */
@keyframes floatParticles {
    from { transform: translateY(100vh) scale(0.3); opacity: 0.2; }
    to { transform: translateY(-10vh) scale(1); opacity: 0.8; }
}
.stApp::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: radial-gradient(circle, rgba(212,175,55,0.15) 1px, transparent 1px);
    background-size: 80px 80px;
    animation: floatParticles 60s linear infinite;
    opacity: 0.2;
    pointer-events: none;
}

/* === Luxury Header === */
.app-header {
    text-align: center;
    margin-bottom: 3rem;
}
.app-header .title-icon {
    font-size: 4rem;
    color: var(--primary-gold);
    text-shadow: 0 0 25px var(--gold-glow), 0 0 60px var(--primary-gold);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { text-shadow: 0 0 25px var(--gold-glow), 0 0 40px var(--primary-gold); }
    50% { text-shadow: 0 0 40px var(--gold-glow), 0 0 80px var(--gold-hover); }
}
.app-header h1 {
    font-size: 3.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, var(--primary-gold), var(--gold-hover));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header p {
    font-size: 1.15rem;
    color: var(--text-secondary);
    max-width: 650px;
    margin: 0 auto;
    font-weight: 300;
    line-height: 1.6;
}

/* === Feature Cards === */
.feature-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.4s ease;
    backdrop-filter: blur(15px);
}
.feature-card:hover {
    transform: translateY(-8px);
    border-color: var(--primary-gold);
    box-shadow: 0 0 25px rgba(212,175,55,0.25);
}
.feature-icon {
    font-size: 2.2rem;
    color: var(--primary-gold);
    margin-bottom: 0.5rem;
}

/* === Input Card === */
.input-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 2.2rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.6);
    transition: all 0.3s ease;
}
.input-card:hover {
    border-color: var(--gold-hover);
    box-shadow: 0 20px 50px rgba(0,0,0,0.8), 0 0 20px var(--gold-glow);
}

/* === Predict Button === */
.stButton>button {
    font-family: var(--font-family);
    background: linear-gradient(135deg, var(--primary-gold), var(--gold-hover));
    color: #111;
    font-weight: 600;
    font-size: 1.15rem;
    padding: 1rem 2rem;
    border-radius: 14px;
    border: none;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 6px 20px var(--gold-glow);
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 30px var(--gold-glow), 0 0 20px var(--primary-gold);
}
.stButton>button:active {
    transform: scale(0.96);
}

/* === Result Card === */
.result-container {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.6);
}
#predicted-disease {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--primary-gold), var(--gold-hover));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
#suggestion-list li {
    background: rgba(255,255,255,0.05);
    border-left: 3px solid var(--primary-gold);
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.6rem;
    color: var(--text-primary);
    transition: transform 0.2s ease;
}
#suggestion-list li:hover {
    transform: translateX(8px);
    background: rgba(255,255,255,0.08);
}

/* === Disclaimer & Footer === */
.disclaimer {
    font-size: 0.9rem;
    color: var(--text-secondary);
    text-align: center;
    padding: 1rem;
    margin-top: 2rem;
    border-radius: 12px;
    border-top: 1px solid var(--card-border);
    background: rgba(30, 30, 30, 0.6);
}
.footer {
    text-align: center;
    color: #777;
    font-size: 0.85rem;
    padding-top: 3rem;
}
</style>
""", unsafe_allow_html=True)
# Load Model and Data
# ----------------------------
@st.cache_data
def load_data():
    """Loads model, medications, and symptoms list with error handling."""
    try:
        with open("disease_predictor.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Fatal Error: 'disease_predictor.pkl' not found. The application cannot start.")
        st.stop()
    except Exception as e:
        st.error(f"Fatal Error loading model: {e}")
        st.stop()

    try:
        medications_df = pd.read_csv("medications.csv")
    except FileNotFoundError:
        st.error("Fatal Error: 'medications.csv' not found. The application cannot start.")
        st.stop()

    try:
        train_df = pd.read_csv("Training.csv").drop(columns=["Unnamed: 133"], errors='ignore')
        symptoms = train_df.drop("prognosis", axis=1).columns.tolist()
    except FileNotFoundError:
        st.error("Fatal Error: 'Training.csv' not found. The application cannot start.")
        st.stop()

    return model, medications_df, symptoms

# Load all necessary data
model, medications_df, symptoms = load_data()

# ----------------------------
# Header Section
# ----------------------------
st.markdown(
    """
    <div class='app-header'>
        <div class='title-icon'>🩺</div>
        <h1>DiagnoX AI</h1>
        <p>Your personal AI health companion for intelligent symptom analysis and preliminary insights.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# NEW: Features Section
# ----------------------------
st.markdown("<div class='features-section'>", unsafe_allow_html=True)
cols = st.columns(3, gap="large")
with cols[0]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">✨</div>
        <div class="feature-title">AI-Powered Analysis</div>
        <div class="feature-description">Leverages a sophisticated machine learning model to analyze your symptoms against a vast dataset of medical information.</div>
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡️</div>
        <div class="feature-title">Instant Results</div>
        <div class="feature-description">Receive immediate, potential health insights based on the symptoms you provide, helping you understand possible conditions quickly.</div>
    </div>
    """, unsafe_allow_html=True)
with cols[2]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧑‍⚕️</div>
        <div class="feature-title">Actionable Guidance</div>
        <div class="feature-description">Provides relevant suggestions and next steps for the predicted condition, empowering you to make informed health decisions.</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Input Section
# ----------------------------
main_cols = st.columns([1, 1.5, 1])
with main_cols[1]: # Center column for input
    with st.container():
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; font-weight: 600;'>Symptom Analysis</h2>", unsafe_allow_html=True)
        selected_symptoms = st.multiselect(
            label="Select the symptoms you are experiencing. You may choose multiple.",
            options=symptoms,
            help="Begin typing to search for a specific symptom.",
            label_visibility="collapsed"
        )
        st.write("") # Spacer
        predict_btn = st.button("Analyze Symptoms", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Prediction & Output Logic
# ----------------------------
if predict_btn:
    st.markdown("---")
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom for analysis.")
    else:
        # Prepare input data for the model
        input_data = [0] * len(symptoms)
        for symptom in selected_symptoms:
            if symptom in symptoms:
                input_data[symptoms.index(symptom)] = 1
        
        input_data = np.array(input_data).reshape(1, -1)

        # Prediction and result display
        try:
            prediction = model.predict(input_data)[0]
            
            # Fetch suggestions from the medications dataframe
            suggestion_row = medications_df[medications_df["Disease"].str.lower() == prediction.lower()]
            suggestions = suggestion_row["Suggestion"].tolist() if not suggestion_row.empty else []

            # Display results in the custom card
            result_cols = st.columns([0.5, 2, 0.5])
            with result_cols[1]:
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                
                res_layout = st.columns([1, 1])
                with res_layout[0]:
                    st.markdown("<div class='result-header'>Potential Condition</div>", unsafe_allow_html=True)
                    st.markdown(f"<h3 id='predicted-disease'>{prediction}</h3>", unsafe_allow_html=True)
                
                with res_layout[1]:
                    st.markdown("<div class='result-header'>Recommended Actions & Insights</div>", unsafe_allow_html=True)
                    if suggestions:
                        suggestion_html = "<ul id='suggestion-list'>"
                        for s in suggestions:
                           suggestion_html += f"<li>{s}</li>"
                        suggestion_html += "</ul>"
                        st.markdown(suggestion_html, unsafe_allow_html=True)
                    else:
                        st.info("No specific actions or medications found in our database for this condition. Please consult a healthcare professional.")

                st.markdown("<br><div class='disclaimer'><strong>Disclaimer:</strong> This is an AI-generated analysis and not a substitute for professional medical advice. Please consult a doctor for an accurate diagnosis and treatment plan.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
else:
    # Initial instruction text
    st.info("👆 Begin by selecting your symptoms in the card above and click 'Analyze Symptoms' to receive your preliminary health insights.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("<p class='footer'>DiagnoX AI &copy; 2025 | Made with ❤️ by Vansh</p>", unsafe_allow_html=True)

