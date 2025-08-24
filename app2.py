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
/* === DiagnoX AI - Sovereign Dark Gold (Lite) === */
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

/* === Background Gradient (smooth only) === */
.stApp {
    font-family: var(--font-family);
    background: linear-gradient(135deg, #0a0a0a, #141414, #1a1a1a, #0f0f0f);
    background-size: 400% 400%;
    animation: auroraShift 25s ease infinite;
    color: var(--text-primary);
}
@keyframes auroraShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* === Header === */
.app-header {
    text-align: center;
    margin-bottom: 2.5rem;
}
.app-header .title-icon {
    font-size: 3.5rem;
    color: var(--primary-gold);
    text-shadow: 0 0 25px var(--gold-glow);
}
.app-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    background: linear-gradient(90deg, var(--primary-gold), var(--gold-hover));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header p {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 620px;
    margin: 0 auto;
}

/* === Cards === */
.feature-card, .input-card, .result-container {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}
.feature-card:hover, .input-card:hover, .result-container:hover {
    border-color: var(--primary-gold);
    box-shadow: 0 0 25px var(--gold-glow);
}
.feature-icon {
    font-size: 2rem;
    color: var(--primary-gold);
    margin-bottom: 0.5rem;
}

/* === Predict Button === */
.stButton>button {
    background: linear-gradient(135deg, var(--primary-gold), var(--gold-hover));
    color: #111;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 0.9rem 2rem;
    border-radius: 12px;
    border: none;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 6px 20px var(--gold-glow);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px var(--gold-glow);
}

/* === Result === */
#predicted-disease {
    font-size: 2.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--primary-gold), var(--gold-hover));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
#suggestion-list li {
    border-left: 3px solid var(--primary-gold);
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.05);
}

/* === Footer === */
.disclaimer {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-align: center;         /* center text */
    padding: 1rem;
    border-top: 1px solid var(--card-border);
    margin-top: 2rem;

    position: fixed;            /* stick to bottom */
    bottom: 0;
    left: 0;
    width: 100%;
    background: rgba(10,10,10,0.9);  /* subtle dark strip */
    backdrop-filter: blur(6px);
    z-index: 999;
}

/* === Made With Line Footer === */
.footer {
        text-align: center;
    padding: 1rem 0;

    color: var(--primary-gold, #D4AF37);

    margin-top: 2rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--primary-gold, #D4AF37);
    letter-spacing: 1px;

    background: transparent;  /* or rgba(0,0,0,0.6) if you want strip */
    border-top: 1px solid rgba(212, 175, 55, 0.4);
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





