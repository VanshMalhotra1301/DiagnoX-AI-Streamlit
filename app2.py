import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Configuration ---
# Set page config once at the beginning
st.set_page_config(
    page_title="DiagnoX AI",
    page_icon="⚜️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Sovereign Gold Theme by Aura Health ---
st.markdown("""
    <style>
        /* === Aura Health - Sovereign Gold v1.0 === */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        :root {
            --primary-gold: #D4AF37; /* A rich, classic gold */
            --primary-gold-hover: #E5C100; /* A brighter gold for interactions */
            --primary-glow: rgba(212, 175, 55, 0.25);
            
            --bg-main-start: #1a1a1a; /* Deep charcoal */
            --bg-main-end: #000000;   /* Pure black for depth */
            
            --card-bg: rgba(20, 20, 20, 0.6); /* Dark, semi-transparent card */
            --card-border: rgba(212, 175, 55, 0.2); /* Subtle gold border */
            --card-shadow: rgba(0, 0, 0, 0.5);
            
            --text-primary: #F0F0F0; /* Off-white for readability */
            --text-secondary: #a0a0a0; /* Grey for subtitles and secondary info */
            --font-family: 'Inter', sans-serif;
        }

        /* --- General Body & App Styling --- */
        body, .stApp {
            font-family: var(--font-family);
            background-image: radial-gradient(circle at top right, var(--bg-main-start), var(--bg-main-end) 80%);
            color: var(--text-primary);
        }

        /* --- Custom Header --- */
        .app-header {
            text-align: center;
            margin-bottom: 4rem;
        }
        .app-header .title-icon {
            font-size: 4rem;
            color: var(--primary-gold);
            text-shadow: 0 0 30px var(--primary-glow), 0 0 50px var(--primary-gold);
        }
        .app-header h1 {
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: -1px;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        .app-header p {
            font-size: 1.15rem;
            color: var(--text-secondary);
            max-width: 600px;
            margin: 0 auto;
            font-weight: 300;
        }

        /* --- Input Card --- */
        .input-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2rem 2.5rem;
            box-shadow: 0 15px 30px var(--card-shadow);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            transition: all 0.3s ease;
            margin-bottom: 2rem;
        }
        .input-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px var(--card-shadow), 0 0 15px var(--primary-glow);
            border-color: rgba(212, 175, 55, 0.4);
        }

        /* --- Predict Button --- */
        .stButton>button {
            font-family: var(--font-family);
            background-image: linear-gradient(45deg, var(--primary-gold), var(--primary-gold-hover));
            color: #101010;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 0.9rem 2rem;
            border-radius: 12px;
            border: none;
            width: 100%;
            cursor: pointer;
            transition: all 0.3s ease-out;
            box-shadow: 0 5px 15px var(--primary-glow);
        }
        .stButton>button:hover {
            transform: scale(1.03);
            box-shadow: 0 8px 25px var(--primary-glow), 0 0 10px var(--primary-gold);
        }

        /* --- Result Display Card --- */
        .result-container {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 15px 30px var(--card-shadow);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }
        .result-header {
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        #predicted-disease {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--primary-gold);
            text-shadow: 0 0 15px var(--primary-glow);
        }
        #suggestion-list {
            list-style-type: none;
            padding-left: 0;
        }
        #suggestion-list li {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.75rem;
            border-left: 3px solid var(--primary-gold);
            font-weight: 300;
            line-height: 1.6;
        }

        /* --- Disclaimer & Footer --- */
        .disclaimer {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-align: center;
            background: rgba(30, 30, 30, 0.5);
            padding: 1rem;
            border-radius: 12px;
            border-top: 1px solid var(--card-border);
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.8rem;
            padding-top: 4rem;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
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
        <div class='title-icon'>⚜️</div>
        <h1>Diagnox AI</h1>
        <p>Your personal AI health companion for intelligent symptom analysis and preliminary insights.</p>
    </div>
    """,
    unsafe_allow_html=True
)

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
st.markdown("<p class='footer'>DiagnoX AI &copy; 2025 | Developed by Vansh</p>", unsafe_allow_html=True)
