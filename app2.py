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
        /* === DiagnoX AI - Celestial Light v1.0 === */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --primary-accent: #007AFF;    /* Professional Blue */
            --secondary-accent: #FF8C66; /* Soft Coral */
            
            --bg-color: #E0E5EC; /* Light, soft background */
            
            --text-primary: #2c3e50;
            --text-secondary: #7f8c8d;
            --font-family: 'Inter', sans-serif;
        }

        /* --- Animated Gradient Background (very subtle) --- */
        @keyframes subtleAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stApp {
            font-family: var(--font-family);
            background: linear-gradient(-45deg, #F0F2F5, #E6E9EF, #F5F7FA, #E8F0F5);
            background-size: 400% 400%;
            animation: subtleAnimation 30s ease infinite;
            color: var(--text-primary);
        }

        /* --- Header --- */
        .app-header {
            text-align: center;
            margin-bottom: 4rem;
        }
        .app-header h1 {
            font-size: 3.8rem;
            font-weight: 700;
            letter-spacing: -2px;
            background: -webkit-linear-gradient(45deg, var(--primary-accent), #2c3e50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .app-header p {
            font-size: 1.2rem;
            color: var(--text-secondary);
            max-width: 700px;
            margin: 0 auto;
            font-weight: 400;
        }
        
        /* --- Neumorphic / Claymorphic Card Style --- */
        .clay-card {
            background: var(--bg-color);
            border-radius: 30px;
            padding: 2.5rem;
            transition: all 0.4s ease;
            height: 100%;
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 
                -10px -10px 20px rgba(255, 255, 255, 0.7), 
                10px 10px 20px rgba(163, 177, 198, 0.6);
        }
        .clay-card:hover {
            transform: translateY(-5px);
            box-shadow: 
                -15px -15px 30px rgba(255, 255, 255, 0.8), 
                15px 15px 30px rgba(163, 177, 198, 0.7);
        }

        .card-header {
            font-size: 1.8rem; font-weight: 600; margin-bottom: 2rem; text-align: center;
        }

        /* --- Button Style --- */
        .stButton>button {
            font-family: var(--font-family);
            background: var(--primary-accent);
            color: #FFFFFF; font-weight: 600; font-size: 1.1rem; padding: 1rem 2rem;
            border-radius: 12px; border: none; width: 100%; cursor: pointer;
            transition: all 0.3s ease-out;
            box-shadow: 5px 5px 10px #bec8d2, -5px -5px 10px #fff;
        }
        .stButton>button:hover {
            background: #0062CC;
            box-shadow: 3px 3px 8px #bec8d2, -3px -3px 8px #fff;
        }
        .stButton>button:active {
            transform: scale(0.98);
            box-shadow: inset 2px 2px 5px #bec8d2, inset -2px -2px 5px #fff;
        }
        
        /* --- Streamlit Multiselect Box Styling --- */
        .stMultiSelect > div > div {
            border-radius: 12px !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            box-shadow: inset 2px 2px 5px #bec8d2, inset -2px -2px 5px #fff;
            background-color: var(--bg-color);
        }

        /* --- Result Display --- */
        .result-header {
            font-weight: 600; color: var(--text-secondary); text-transform: uppercase;
            letter-spacing: 1px; font-size: 0.9rem; margin-bottom: 0.5rem; text-align: center;
        }
        #predicted-disease {
            font-size: 2.6rem; font-weight: 700; text-align: center;
            color: var(--primary-accent);
            margin-bottom: 2rem;
        }
        #suggestion-list li {
            background-color: #E8F0F5; padding: 1rem; border-radius: 12px;
            margin-bottom: 0.75rem; border-left: 4px solid var(--secondary-accent);
            font-weight: 400; line-height: 1.6;
        }

        /* --- "How It Works" Section --- */
        .info-section { padding: 2.5rem; margin-top: 2rem; }
        .info-title { font-size: 1.5rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.75rem; }
        .info-icon { font-size: 2rem; margin-right: 1rem; vertical-align: middle; }
        .info-text { color: var(--text-secondary); font-weight: 400; line-height: 1.7; }

        /* --- Footer --- */
        .footer { text-align: center; color: #aaa; font-size: 0.9rem; padding-top: 5rem; }
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
