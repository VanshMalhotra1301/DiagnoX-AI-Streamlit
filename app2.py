import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="DiagnoX AI | Cosmic Symptom Analysis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- NEW: Cosmic Glow Theme ---
st.markdown("""
    <style>
        /* === DiagnoX AI - Cosmic Glow v2.0 === */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        :root {
            --primary-accent: #FF00A9;  /* Vibrant Magenta */
            --secondary-accent: #00F2FF; /* Electric Cyan */
            --primary-glow: rgba(255, 0, 169, 0.4);
            
            --bg-main-start: #0a0118;
            --bg-main-end: #180033;
            
            --card-bg: rgba(10, 5, 30, 0.6); /* Deeper, semi-transparent purple */
            --card-border: rgba(255, 0, 169, 0.2);
            --card-shadow: rgba(0, 0, 0, 0.6);
            
            --text-primary: #F0F2F6;
            --text-secondary: #A0AEC0;
            --font-family: 'Poppins', sans-serif;
        }

        /* --- Animated Gradient Background --- */
        @keyframes cosmicAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stApp {
            font-family: var(--font-family);
            background: linear-gradient(-45deg, #0a0118, #180033, #3c004a, #0d1117);
            background-size: 400% 400%;
            animation: cosmicAnimation 25s ease infinite;
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
            background: -webkit-linear-gradient(45deg, var(--primary-accent), var(--secondary-accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .app-header p {
            font-size: 1.2rem;
            color: var(--text-secondary);
            max-width: 700px;
            margin: 0 auto;
            font-weight: 300;
        }
        
        /* --- Glassmorphism Card with Gradient Border --- */
        .glass-card {
            background: var(--card-bg);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 15px 30px var(--card-shadow);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            transition: all 0.4s ease;
            height: 100%;
            border: 1px solid transparent;
            background-clip: padding-box;
            position: relative;
        }
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0; right: 0; bottom: 0; left: 0;
            z-index: -1;
            margin: -1px; /* Border width */
            border-radius: inherit; /* Follow the parent's border-radius */
            background: linear-gradient(135deg, var(--primary-accent), var(--secondary-accent));
            opacity: 0.2;
            transition: opacity 0.4s ease;
        }
        .glass-card:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 25px 50px var(--card-shadow), 0 0 40px var(--primary-glow);
        }
        .glass-card:hover::before {
            opacity: 0.8;
        }

        .card-header {
            font-size: 1.8rem; font-weight: 600; margin-bottom: 1.5rem; text-align: center;
        }

        /* --- Button --- */
        .stButton>button {
            font-family: var(--font-family);
            background-image: linear-gradient(90deg, var(--primary-accent) 0%, var(--secondary-accent) 100%);
            color: #FFFFFF; font-weight: 600; font-size: 1.1rem; padding: 1rem 2rem;
            border-radius: 12px; border: none; width: 100%; cursor: pointer;
            transition: all 0.3s ease-out;
            box-shadow: 0 5px 20px var(--primary-glow);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 30px var(--primary-glow), 0 0 20px var(--secondary-accent);
        }

        /* --- Result Display --- */
        .result-header {
            font-weight: 600; color: var(--text-secondary); text-transform: uppercase;
            letter-spacing: 1px; font-size: 0.9rem; margin-bottom: 0.5rem;
        }
        #predicted-disease {
            font-size: 2.6rem; font-weight: 700;
            background: -webkit-linear-gradient(45deg, var(--primary-accent), var(--secondary-accent));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-shadow: 0 0 25px var(--primary-glow); margin-bottom: 2rem;
        }
        #suggestion-list li {
            background-color: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 10px;
            margin-bottom: 0.75rem; border-left: 4px solid var(--secondary-accent);
            font-weight: 300; line-height: 1.6; transition: all 0.2s ease;
        }
        #suggestion-list li:hover {
            background-color: rgba(0, 242, 255, 0.15); transform: translateX(5px);
        }

        /* --- "How It Works" Section --- */
        .info-section {
            padding: 2.5rem;
            margin-top: 2rem;
        }
        .info-title {
            font-size: 1.5rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.75rem;
        }
        .info-icon {
            font-size: 2rem; margin-right: 1rem; vertical-align: middle;
        }
        .info-text {
            color: var(--text-secondary); font-weight: 300; line-height: 1.7;
        }

        /* --- Footer --- */
        .footer { text-align: center; color: #666; font-size: 0.8rem; padding-top: 5rem; }
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
        <p>Harnessing the power of machine learning to provide preliminary health insights from your symptoms.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Main Layout (Input & Output)
# ----------------------------
col1, col2 = st.columns([1.2, 1], gap="large")

# --- INPUT COLUMN ---
with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-header'>Symptom Analysis Matrix</h2>", unsafe_allow_html=True)
    
    selected_symptoms = st.multiselect(
        label="Select your symptoms:",
        options=symptoms,
        help="Type to search and select multiple symptoms.",
        label_visibility="collapsed",
        placeholder="e.g., 'fever', 'headache', 'cough'..."
    )
    st.write("") # Spacer
    predict_btn = st.button("🔮 Initiate Diagnosis AI", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- OUTPUT COLUMN ---
with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    if predict_btn:
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom for analysis.")
        else:
            input_data = [0] * len(symptoms)
            for symptom in selected_symptoms:
                if symptom in symptoms:
                    input_data[symptoms.index(symptom)] = 1
            
            try:
                prediction = model.predict(np.array(input_data).reshape(1, -1))[0]
                suggestion_row = medications_df[medications_df["Disease"].str.lower() == prediction.lower()]
                suggestions = suggestion_row["Suggestion"].tolist() if not suggestion_row.empty else []

                st.markdown("<div class='result-header'>Primary Analysis</div>", unsafe_allow_html=True)
                st.markdown(f"<h3 id='predicted-disease'>{prediction}</h3>", unsafe_allow_html=True)
                
                st.markdown("<div class='result-header'>💡 Recommended Actions</div>", unsafe_allow_html=True)
                if suggestions:
                    st.markdown("".join([f"<ul id='suggestion-list'>{''.join([f'<li>{s}</li>' for s in suggestions])}</ul>"]), unsafe_allow_html=True)
                else:
                    st.info("No specific actions found. Please consult a healthcare professional.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Your analysis results will appear here after you input symptoms and initiate the AI.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- NEW: "How It Works" Section ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='glass-card info-section'>
        <h2 class='card-header' style='margin-bottom: 2rem;'>Behind the AI</h2>
        <div style='display: flex; gap: 2rem;'>
            <div style='flex: 1;'>
                <h3 class='info-title'><span class='info-icon'>🧠</span>The Model</h3>
                <p class='info-text'>DiagnoX AI utilizes a sophisticated <strong>Machine Learning classifier</strong>. This model has been trained to recognize complex patterns and correlations between various symptoms and a wide range of medical conditions.</p>
            </div>
            <div style='flex: 1;'>
                <h3 class='info-title'><span class='info-icon'>📊</span>The Dataset</h3>
                <p class='info-text'>The AI was trained on a comprehensive, anonymized dataset containing thousands of patient cases. This dataset maps <strong>132 distinct symptoms</strong> to <strong>41 different diseases</strong>, allowing the model to learn from a vast repository of medical knowledge.</p>
            </div>
            <div style='flex: 1;'>
                <h3 class='info-title'><span class='info-icon'>⚙️</span>The Process</h3>
                <p class='info-text'>When you input your symptoms, the AI converts them into a numerical format. It then processes this data, comparing your unique symptom profile against the patterns it learned during training to predict the most probable condition.</p>
            </div>
        </div>
        <p class='info-text' style='text-align: center; margin-top: 2rem; font-size: 0.9rem; color: var(--text-secondary);'>
            <strong>Disclaimer:</strong> This tool provides preliminary, AI-generated insights and is not a substitute for professional medical advice. Always consult a qualified doctor for an accurate diagnosis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<p class='footer'>DiagnoX AI &copy; 2025 | Vansh</p>", unsafe_allow_html=True)
