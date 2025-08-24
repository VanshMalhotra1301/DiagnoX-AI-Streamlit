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

        /* Header */
        .app-header {
            text-align: center;
            margin-bottom: 3rem;
        }
        .app-header .logo {
            font-size: 3rem;
            color: var(--primary-color);
            text-shadow: 0 0 25px var(--primary-color);
        }
        .app-header h1 {
            font-size: 2.75rem;
            font-weight: 800;
            background: linear-gradient(45deg, var(--text-primary), #bdc3c7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .app-header p {
            color: var(--text-secondary);
            font-size: 1.15rem;
            line-height: 1.6;
        }

        /* Card */
        .glass-card {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            border: 1px solid var(--card-border);
            margin-bottom: 2rem;
        }

        /* Button */
        .stButton>button {
            background: var(--primary-color);
            color: #0f172a;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 0.8rem 1.5rem;
            border-radius: 12px;
            border: none;
            width: 100%;
            cursor: pointer;
        }
        .stButton>button:hover {
            background: var(--primary-hover);
            box-shadow: 0 8px 15px var(--primary-glow);
        }

        /* Result Card */
        .result-card {
            padding: 1.5rem;
            border-radius: 1.25rem;
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(14px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            margin-top: 1.5rem;
        }
        .result-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        #predicted-disease {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
            text-shadow: 0 0 20px var(--primary-glow);
            margin: 1rem 0;
        }
        .suggestion-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-secondary);
        }
        #suggestion {
            font-size: 1.1rem;
            line-height: 1.6;
            color: var(--text-primary);
        }

        /* Disclaimer */
        .disclaimer {
            margin-top: 1.5rem;
            font-size: 0.85rem;
            color: #fbbf24;
            background: rgba(251, 191, 36, 0.1);
            border-left: 3px solid #fbbf24;
            padding: 0.75rem;
            border-radius: 0.75rem;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ✅ Page config
st.set_page_config(page_title="DiagnoX AI", page_icon="🩺", layout="wide")

# ----------------------------
# Load model and data
# ----------------------------
@st.cache_data
def load_data():
    # Load ML model
    try:
        with open("disease_predictor.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception:
        st.error("❌ Error loading model. Ensure 'disease_predictor.pkl' exists and is valid.")
        st.stop()

    # Load medications CSV
    try:
        medications_df = pd.read_csv("medications.csv")
    except Exception:
        st.error("❌ Error loading 'medications.csv'. Ensure the file exists in the app directory.")
        st.stop()

    # Load training data (symptoms list)
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
# Streamlit UI
# ----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #FDB813;'>🩺 Diagnox AI</h1>
    <p style='text-align: center; color: #cbd5e1;'>
    Select your symptoms and get possible disease predictions with medical suggestions.
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Centered Input Section
with st.container():
    st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
    col = st.columns([1,2,1])[1]  # center column
    with col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("⚡ Input Symptoms")
        selected_symptoms = st.multiselect(
            "Choose symptoms you are experiencing:",
            options=symptoms,
            help="Start typing to search symptoms.",
        )

        predict_btn = st.button("🔍 Predict", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Prediction Logic
# ----------------------------
if predict_btn:
    if not selected_symptoms:
        st.warning("⚠ Please select at least one symptom before predicting.")
    else:
        # Convert symptoms to input format
        input_data = [0] * len(symptoms)
        for s in selected_symptoms:
            input_data[symptoms.index(s)] = 1
        input_data = np.array(input_data).reshape(1, -1)

        # Predict
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"✅ Predicted Disease: *{prediction}*")

            # ✅ Match with Disease column in medications.csv
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
    st.info("👆 Select symptoms above and click *Predict*.")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray;'>Built with ❤ By Vansh</p>
    """,
    unsafe_allow_html=True,
)
