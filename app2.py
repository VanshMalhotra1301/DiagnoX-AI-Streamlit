import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Configuration ---
# Set page config once at the beginning
st.set_page_config(
    page_title="DiagnoX AI | Advanced Symptom Analysis",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Aurora Gradient & Glassmorphism Theme for DiagnoX AI ---
st.markdown("""
    <style>
        /* === DiagnoX AI - Aurora Gradient & Glassmorphism v1.0 === */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        :root {
            --primary-accent: #00A9FF;
            --primary-accent-hover: #00E0FF;
            --secondary-accent: #8900FF;
            --primary-glow: rgba(0, 169, 255, 0.4);
            
            --bg-main-start: #0D1117;
            --bg-main-end: #000000;
            
            --card-bg: rgba(15, 23, 42, 0.6); /* Semi-transparent dark blue */
            --card-border: rgba(0, 169, 255, 0.2);
            --card-shadow: rgba(0, 0, 0, 0.5);
            
            --text-primary: #F0F2F6;
            --text-secondary: #A0AEC0;
            --font-family: 'Poppins', sans-serif;
        }

        /* --- Keyframe Animation for Background --- */
        @keyframes gradientAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* --- General Body & App Styling --- */
        body, .stApp {
            font-family: var(--font-family);
            background: linear-gradient(-45deg, #0d1117, #1e0033, #001f3f, #0d1117);
            background-size: 400% 400%;
            animation: gradientAnimation 25s ease infinite;
            color: var(--text-primary);
        }

        /* --- Custom Header --- */
        .app-header {
            text-align: center;
            margin-bottom: 4rem;
        }
        .app-header h1 {
            font-size: 3.5rem;
            font-weight: 700;
            letter-spacing: -2px;
            background: -webkit-linear-gradient(45deg, var(--primary-accent), var(--secondary-accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .app-header p {
            font-size: 1.15rem;
            color: var(--text-secondary);
            max-width: 650px;
            margin: 0 auto;
            font-weight: 300;
        }
        
        /* --- Main Content Cards (Glassmorphism) --- */
        .glass-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 15px 30px var(--card-shadow);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            transition: all 0.3s ease;
            height: 100%; /* For equal height columns */
        }
        .glass-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px var(--card-shadow), 0 0 25px var(--primary-glow);
            border-color: rgba(0, 169, 255, 0.5);
        }

        /* --- Input Section --- */
        .input-header {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        /* --- Predict Button --- */
        .stButton>button {
            font-family: var(--font-family);
            background-image: linear-gradient(90deg, var(--primary-accent), var(--secondary-accent));
            color: #FFFFFF;
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
            transform: scale(1.05);
            box-shadow: 0 8px 25px var(--primary-glow), 0 0 15px var(--primary-accent);
        }
        .stButton>button:active {
            transform: scale(0.98);
        }

        /* --- Result Display Card --- */
        .result-header {
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        #predicted-disease {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-accent-hover);
            text-shadow: 0 0 20px var(--primary-glow);
            margin-bottom: 2rem;
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
            border-left: 4px solid var(--primary-accent);
            font-weight: 300;
            line-height: 1.6;
            transition: all 0.2s ease;
        }
        #suggestion-list li:hover {
            background-color: rgba(0, 169, 255, 0.15);
            transform: translateX(5px);
        }
        
        /* --- Placeholder Text --- */
        .placeholder-text {
            font-size: 1.1rem;
            font-weight: 300;
            color: var(--text-secondary);
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }

        /* --- Disclaimer & Footer --- */
        .disclaimer-box {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-align: center;
            background: rgba(10, 10, 10, 0.3);
            padding: 1rem;
            border-radius: 12px;
            border-top: 1px solid var(--card-border);
            margin-top: 2rem;
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.8rem;
            padding-top: 5rem;
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
        <h1>DiagnoX AI</h1>
        <p>Your intelligent health companion. Describe your symptoms for an AI-powered preliminary analysis and actionable insights.</p>
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
    st.markdown("<h2 class='input-header'>🔬 Symptom Input</h2>", unsafe_allow_html=True)
    
    selected_symptoms = st.multiselect(
        label="Select the symptoms you are experiencing. You may choose multiple.",
        options=symptoms,
        help="Begin typing to search for a specific symptom.",
        label_visibility="collapsed",
        placeholder="e.g., 'fever', 'headache', 'cough'..."
    )
    
    st.write("") # Spacer
    predict_btn = st.button("✨ Analyze Symptoms", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- OUTPUT COLUMN ---
with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    if predict_btn:
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

                # Display results
                st.markdown("<div class='result-header'>Primary Analysis</div>", unsafe_allow_html=True)
                st.markdown(f"<h3 id='predicted-disease'>{prediction}</h3>", unsafe_allow_html=True)
                
                st.markdown("<div class='result-header'>💡 Recommended Actions & Insights</div>", unsafe_allow_html=True)
                if suggestions:
                    suggestion_html = "<ul id='suggestion-list'>"
                    for s in suggestions:
                        suggestion_html += f"<li>{s}</li>"
                    suggestion_html += "</ul>"
                    st.markdown(suggestion_html, unsafe_allow_html=True)
                else:
                    st.info("No specific actions found in our database. Please consult a healthcare professional for guidance.")

                st.markdown("<div class='disclaimer-box'><strong>Disclaimer:</strong> This is an AI-generated analysis and not a substitute for professional medical advice. Please consult a doctor for an accurate diagnosis.</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
    else:
        # Initial placeholder text
        st.markdown("<div class='placeholder-text'>Your analysis results will appear here.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Footer
# ----------------------------
st.markdown("<p class='footer'>DiagnoX AI &copy; 2025 | Made with ❤️ by Vansh</p>", unsafe_allow_html=True)
