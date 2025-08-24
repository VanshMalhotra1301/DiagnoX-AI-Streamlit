import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="DiagnoX AI | Advanced Symptom Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Enhanced CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

:root {
    --primary-gold: #D4AF37;
    --gold-hover: #FFD700;
    --gold-glow: rgba(212, 175, 55, 0.25);
    --bg-dark-1: #0a0a0a;
    --bg-dark-2: #141414;
    --text-primary: #f0f0f0;
    --text-secondary: #a0a0a0;
    --card-bg: rgba(20, 20, 20, 0.75);
    --card-border: rgba(212, 175, 55, 0.2);
    --font-family-main: 'Poppins', sans-serif;
    --font-family-mono: 'Roboto Mono', monospace;
}

/* === Core App Styling === */
.stApp {
    font-family: var(--font-family-main);
    background: linear-gradient(135deg, var(--bg-dark-1) 0%, #111 50%, var(--bg-dark-2) 100%);
    color: var(--text-primary);
}

/* === Header === */
.app-header {
    text-align: center;
    margin-bottom: 3rem;
}
.app-header .title-icon {
    font-size: 4rem;
    color: var(--primary-gold);
    text-shadow: 0 0 30px var(--gold-glow);
    animation: pulse-icon 2s infinite ease-in-out;
}
@keyframes pulse-icon {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
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
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 650px;
    margin: 0 auto;
}

/* === Cards (Features, Input, Results) === */
.card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 1.75rem;
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    transition: all 0.3s ease;
}
.card:hover {
    border-color: var(--primary-gold);
    box-shadow: 0 0 30px var(--gold-glow);
    transform: translateY(-5px);
}

/* Feature Cards Specifics */
.feature-icon {
    font-size: 2.5rem;
    color: var(--primary-gold);
    margin-bottom: 0.75rem;
}
.feature-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.feature-description {
    font-size: 0.95rem;
    color: var(--text-secondary);
}

/* Input Card Specifics */
.input-card h2 {
    text-align: center;
    font-weight: 600;
    color: var(--text-primary);
}

/* === Predict Button === */
.stButton>button {
    background: linear-gradient(135deg, var(--primary-gold), #B8860B);
    color: #0a0a0a;
    font-weight: 700;
    font-size: 1.2rem;
    padding: 1rem 2.5rem;
    border-radius: 15px;
    border: none;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px var(--gold-glow);
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 12px 30px rgba(212, 175, 55, 0.4);
}

/* === Result Section === */
.result-container {
    padding: 2rem;
    margin-top: 1rem;
}
.result-header {
    font-size: 1.1rem;
    font-weight: 400;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}
.predicted-disease-container {
    font-family: var(--font-family-mono);
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--primary-gold), var(--gold-hover));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    min-height: 50px; /* Reserve space for typewriter */
}
.suggestion-list li {
    border-left: 3px solid var(--primary-gold);
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.75rem;
    background: rgba(255, 255, 255, 0.05);
}
.disclaimer-box {
    font-size: 0.9rem;
    color: var(--text-secondary);
    text-align: center;
    padding: 1rem;
    border-top: 1px solid var(--card-border);
    margin-top: 1.5rem;
    background: rgba(10,10,10,0.5);
    border-radius: 0 0 18px 18px;
}

/* Confidence Gauge */
.confidence-section { margin-top: 1rem; }
.confidence-gauge {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    overflow: hidden;
    height: 25px;
    border: 1px solid var(--card-border);
}
.confidence-bar {
    background: linear-gradient(90deg, #B8860B, var(--primary-gold));
    height: 100%;
    border-radius: 8px;
    transition: width 1.5s ease-in-out;
}
.confidence-label {
    text-align: right;
    font-family: var(--font-family-mono);
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* === Analysis "Thinking" Animation === */
.thinking-container {
    text-align: center;
    padding: 2rem;
}
.thinking-text {
    font-family: var(--font-family-mono);
    color: var(--primary-gold);
    margin-bottom: 1rem;
}
.loader {
    width: 80px;
    height: 80px;
    border: 5px solid var(--card-border);
    border-top-color: var(--primary-gold);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 1rem auto;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}

/* === Footer === */
.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    font-size: 0.9rem;
    color: var(--text-secondary);
    border-top: 1px solid var(--card-border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads model, medications, and symptoms list with robust error handling."""
    try:
        with open("disease_predictor.pkl", "rb") as f:
            model = pickle.load(f)
        medications_df = pd.read_csv("medications.csv")
        train_df = pd.read_csv("Training.csv").drop(columns=["Unnamed: 133"], errors='ignore')
        symptoms = sorted(train_df.drop("prognosis", axis=1).columns.tolist())
        return model, medications_df, symptoms
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A required file was not found: {e.filename}. The application cannot start.")
        st.stop()
    except Exception as e:
        st.error(f"Fatal Error during data loading: {e}")
        st.stop()

model, medications_df, symptoms = load_data()

# --- Initialize Session State ---
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probability = None
    st.session_state.suggestions = None
    st.session_state.selected_symptoms = None

# --- Helper Functions ---
def typewriter(text: str, speed: float):
    """Displays text with a typewriter effect."""
    container = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        container.markdown(f"<span class='predicted-disease-container'>{displayed_text}▌</span>", unsafe_allow_html=True)
        time.sleep(speed)
    container.markdown(f"<span class='predicted-disease-container'>{displayed_text}</span>", unsafe_allow_html=True)

# --- UI Rendering Functions ---

def render_header():
    st.markdown("""
        <div class='app-header'>
            <div class='title-icon'>🩺</div>
            <h1>DiagnoX AI</h1>
            <p>Your personal AI health companion for intelligent symptom analysis. Input your symptoms to receive instant, data-driven preliminary insights.</p>
        </div>
    """, unsafe_allow_html=True)

def render_feature_cards():
    cols = st.columns(3, gap="large")
    features = [
        {"icon": "✨", "title": "AI-Powered Analysis", "desc": "Leverages a sophisticated machine learning model to analyze your symptoms against a vast dataset of medical information."},
        {"icon": "⚡️", "title": "Instant Results", "desc": "Receive immediate, potential health insights, including a confidence score, to help you understand possible conditions quickly."},
        {"icon": "🧑‍⚕️", "title": "Actionable Guidance", "desc": "Provides relevant suggestions and next steps for the predicted condition, empowering you to make informed health decisions."}
    ]
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div class="card feature-card">
                <div class="feature-icon">{features[i]['icon']}</div>
                <div class="feature-title">{features[i]['title']}</div>
                <div class="feature-description">{features[i]['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

def render_input_form():
    st.markdown("<br>", unsafe_allow_html=True)
    main_cols = st.columns([1, 1.5, 1])
    with main_cols[1]:
        with st.container():
            st.markdown("<div class='card input-card'>", unsafe_allow_html=True)
            st.markdown("<h2>Symptom Analysis Engine</h2>", unsafe_allow_html=True)
            selected_symptoms = st.multiselect(
                label="Select the symptoms you are experiencing.",
                options=symptoms,
                help="Begin typing to search and select multiple symptoms.",
                label_visibility="collapsed"
            )
            st.write("") # Spacer
            if st.button("Analyze Symptoms", use_container_width=True):
                if not selected_symptoms:
                    st.warning("⚠️ Please select at least one symptom for analysis.")
                    st.session_state.prediction = None # Reset state
                else:
                    # Show "thinking" animation
                    with st.spinner(''):
                         st.markdown("""
                            <div class="thinking-container">
                                <div class="loader"></div>
                                <div class="thinking-text">DIAGNOX AI IS ANALYZING...</div>
                            </div>
                        """, unsafe_allow_html=True)
                         time.sleep(2) # Simulate processing time

                    # Prepare input data for the model
                    input_data = [0] * len(symptoms)
                    for symptom in selected_symptoms:
                        if symptom in symptoms:
                            input_data[symptoms.index(symptom)] = 1
                    
                    input_data = np.array(input_data).reshape(1, -1)

                    # --- Prediction Logic ---
                    try:
                        prediction_proba = model.predict_proba(input_data)[0]
                        max_proba_index = np.argmax(prediction_proba)
                        
                        st.session_state.prediction = model.classes_[max_proba_index]
                        st.session_state.probability = prediction_proba[max_proba_index]
                        st.session_state.selected_symptoms = selected_symptoms

                        suggestion_row = medications_df[medications_df["Disease"].str.lower() == st.session_state.prediction.lower()]
                        st.session_state.suggestions = suggestion_row["Suggestion"].tolist() if not suggestion_row.empty else []
                        
                        st.rerun() # Rerun to display results below

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
                        st.session_state.prediction = None # Reset state

            st.markdown("</div>", unsafe_allow_html=True)

def render_results():
    if st.session_state.prediction:
        st.markdown("---")
        result_cols = st.columns([0.5, 2, 0.5])
        with result_cols[1]:
            st.markdown("<div class='card result-container'>", unsafe_allow_html=True)
            
            res_layout = st.columns([1.2, 1])
            with res_layout[0]:
                st.markdown("<div class='result-header'>Potential Condition</div>", unsafe_allow_html=True)
                typewriter(st.session_state.prediction, 0.05)
                
                # Confidence Gauge
                st.markdown("<div class='result-header confidence-section'>Confidence Score</div>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='confidence-gauge'>
                        <div class='confidence-bar' style='width: {st.session_state.probability*100:.2f}%;'></div>
                    </div>
                    <div class='confidence-label'>{st.session_state.probability*100:.2f}%</div>
                """, unsafe_allow_html=True)

            with res_layout[1]:
                st.markdown("<div class='result-header'>Recommended Actions</div>", unsafe_allow_html=True)
                if st.session_state.suggestions:
                    suggestion_html = "<ul class='suggestion-list'>"
                    for s in st.session_state.suggestions:
                        suggestion_html += f"<li>{s}</li>"
                    suggestion_html += "</ul>"
                    st.markdown(suggestion_html, unsafe_allow_html=True)
                else:
                    st.info("No specific actions found. Please consult a healthcare professional for guidance.")

            st.markdown("<div class='disclaimer-box'><strong>Disclaimer:</strong> This is an AI-generated analysis and not a substitute for professional medical advice. Please consult a doctor for an accurate diagnosis.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

def render_footer():
    st.markdown("""
        <div class="footer">
            DiagnoX AI &copy; 2025 | Developed with ❤️ by Vansh
        </div>
    """, unsafe_allow_html=True)


# --- Main App Flow ---
render_header()
render_feature_cards()
render_input_form()

# Display results if a prediction has been made
if st.session_state.prediction:
    render_results()
else:
    # Initial instruction text
    st.info("👆 Begin by selecting your symptoms above and click 'Analyze Symptoms' to receive your preliminary health insights.")

render_footer()
