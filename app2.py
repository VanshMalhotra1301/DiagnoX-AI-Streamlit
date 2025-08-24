import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="DiagnoX AI | Celestial Symptom Analysis",
    page_icon="🕊️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- NEW: Celestial Light Theme ---
st.markdown("""
<style>
/* === DiagnoX AI - Ultra Luxury Celestial Light v2.0 === */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-accent: #6C63FF;   /* Vibrant Violet */
    --secondary-accent: #FF7E5F; /* Lively Coral */
    --tertiary-accent: #43E97B;  /* Neon Mint */

    --bg-light: #F9FAFC;
    --glass-bg: rgba(255, 255, 255, 0.6);

    --text-primary: #1a1a1a;
    --text-secondary: #555;
    --font-family: 'Poppins', 'Inter', sans-serif;
}

/* === Background with Animated Gradient Aurora === */
@keyframes aurora {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp {
    font-family: var(--font-family);
    background: linear-gradient(135deg,
        #f5f7fa,
        #e4ecf7,
        #fafcff,
        #e6f0ff,
        #fff9f5);
    background-size: 400% 400%;
    animation: aurora 40s ease infinite;
    color: var(--text-primary);
}

/* === Luxury Header === */
.app-header h1 {
    font-size: 4rem;
    font-weight: 700;
    letter-spacing: -2px;
    background: linear-gradient(90deg, var(--primary-accent), var(--secondary-accent), var(--tertiary-accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 4px 20px rgba(108, 99, 255, 0.2);
}
.app-header p {
    font-size: 1.25rem;
    color: var(--text-secondary);
    max-width: 720px;
    margin: 0 auto;
    opacity: 0.85;
}

/* === Luxury Glass Cards === */
.clay-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px) saturate(180%);
    border-radius: 28px;
    padding: 2.5rem;
    border: 1px solid rgba(255,255,255,0.35);
    box-shadow: 0 12px 40px rgba(0,0,0,0.08), 0 0 30px rgba(108, 99, 255, 0.12);
    transition: all 0.4s ease;
}
.clay-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 18px 60px rgba(0,0,0,0.12), 0 0 40px rgba(108, 99, 255, 0.2);
}

/* === Button Styling === */
.stButton>button {
    font-family: var(--font-family);
    background: linear-gradient(135deg, var(--primary-accent), var(--secondary-accent));
    color: #fff; font-weight: 600; font-size: 1.15rem;
    padding: 1rem 2rem; border-radius: 14px;
    border: none; cursor: pointer; width: 100%;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 6px 20px rgba(108,99,255,0.3);
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 25px rgba(255,126,95,0.3);
    background: linear-gradient(135deg, var(--secondary-accent), var(--tertiary-accent));
}
.stButton>button:active {
    transform: scale(0.97);
}

/* === Multiselect Box Luxury === */
.stMultiSelect > div > div {
    border-radius: 14px !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    background: rgba(255,255,255,0.8);
    box-shadow: inset 2px 2px 6px rgba(0,0,0,0.05),
                inset -2px -2px 6px rgba(255,255,255,0.9);
}

/* === Prediction Result === */
#predicted-disease {
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, var(--primary-accent), var(--secondary-accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
    text-shadow: 0 6px 25px rgba(108, 99, 255, 0.2);
}
#suggestion-list li {
    background: rgba(255,255,255,0.85);
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    border-left: 4px solid var(--tertiary-accent);
    font-weight: 500;
    transition: transform 0.2s ease;
}
#suggestion-list li:hover {
    transform: translateX(6px);
    background: rgba(255,255,255,0.95);
}

/* === Info Section === */
.info-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--primary-accent);
}
.info-text {
    color: var(--text-secondary);
    line-height: 1.75;
}

/* === Footer === */
.footer {
    text-align: center;
    color: #888;
    font-size: 0.95rem;
    padding-top: 5rem;
}
</style>
""", unsafe_allow_html=True)
# ----------------------------
# Load Model and Data
# ----------------------------
@st.cache_data
def load_data():
    """Loads all necessary files with error handling."""
    try:
        model = pickle.load(open("disease_predictor.pkl", "rb"))
        medications_df = pd.read_csv("medications.csv")
        train_df = pd.read_csv("Training.csv").drop(columns=["Unnamed: 133"], errors='ignore')
        symptoms = train_df.drop("prognosis", axis=1).columns.tolist()
        return model, medications_df, symptoms
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A required file was not found ({e.filename}). The application cannot start.")
        st.stop()
    except Exception as e:
        st.error(f"Fatal Error loading data: {e}")
        st.stop()

model, medications_df, symptoms = load_data()

# ----------------------------
# Header Section
# ----------------------------
st.markdown(
    """
    <div class='app-header'>
        <h1>DiagnoX AI</h1>
        <p>Intelligent Health Insights. Your first step towards understanding your symptoms.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Main Layout (Input & Output)
# ----------------------------
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("<div class='clay-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-header'>Symptom Checker</h2>", unsafe_allow_html=True)
    
    selected_symptoms = st.multiselect(
        label="Enter your symptoms:",
        options=symptoms,
        help="Type to search and select multiple symptoms.",
        label_visibility="collapsed",
        placeholder="e.g., 'fever', 'headache', 'cough'..."
    )
    st.write("") # Spacer
    predict_btn = st.button("Analyze Symptoms", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='clay-card'>", unsafe_allow_html=True)
    if predict_btn:
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom to begin analysis.")
        else:
            input_data = [0] * len(symptoms)
            for symptom in selected_symptoms:
                if symptom in symptoms:
                    input_data[symptoms.index(symptom)] = 1
            
            try:
                prediction = model.predict(np.array(input_data).reshape(1, -1))[0]
                suggestion_row = medications_df[medications_df["Disease"].str.lower() == prediction.lower()]
                suggestions = suggestion_row["Suggestion"].tolist() if not suggestion_row.empty else []

                st.markdown("<div class='result-header'>Preliminary Analysis</div>", unsafe_allow_html=True)
                st.markdown(f"<h3 id='predicted-disease'>{prediction}</h3>", unsafe_allow_html=True)
                
                st.markdown("<div class='result-header'>Recommended Actions</div>", unsafe_allow_html=True)
                if suggestions:
                    st.markdown("".join([f"<ul id='suggestion-list'>{''.join([f'<li>{s}</li>' for s in suggestions])}</ul>"]), unsafe_allow_html=True)
                else:
                    st.info("No specific actions found. Please consult a healthcare professional.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Your analysis will be displayed here.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# How It Works Section
# ----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='clay-card info-section'>
        <h2 class='card-header' style='margin-bottom: 2rem;'>The Technology Behind DiagnoX</h2>
        <div style='display: flex; gap: 2.5rem; text-align: center;'>
            <div style='flex: 1;'>
                <h3 class='info-title'><span class='info-icon'>🧠</span>The Model</h3>
                <p class='info-text'>DiagnoX employs a machine learning classifier trained to recognize complex patterns between symptoms and medical conditions.</p>
            </div>
            <div style='flex: 1;'>
                <h3 class='info-title'><span class='info-icon'>📊</span>The Data</h3>
                <p class='info-text'>The AI was trained on a comprehensive dataset mapping <strong>132 symptoms</strong> to <strong>41 diseases</strong>, learning from thousands of anonymized cases.</p>
            </div>
            <div style='flex: 1;'>
                <h3 class='info-title'><span class='info-icon'>⚙️</span>The Process</h3>
                <p class='info-text'>Your symptoms are converted into a unique profile, which the AI compares against its learned patterns to find the most probable match.</p>
            </div>
        </div>
        <p class='info-text' style='text-align: center; margin-top: 2rem; font-size: 0.9rem; color: var(--text-secondary);'>
            <strong>Disclaimer:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<p class='footer'>DiagnoX AI &copy; 2025 | Built by Vansh</p>", unsafe_allow_html=True)

