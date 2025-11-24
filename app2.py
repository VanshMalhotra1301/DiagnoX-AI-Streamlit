import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time
from datetime import datetime
from fpdf import FPDF
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="DiagnoX AI Pro | Personalized Diagnosis Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Enhanced CSS Styling (includes centered header + themed buttons) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

:root {
    --primary-gold: #FFD700;
    --gold-hover: #FFEA70;
    --gold-glow: rgba(255, 215, 0, 0.35);
    --bg-dark-1: #050505;
    --text-primary: #f5f5f5;
    --text-secondary: #b5b5b5;
    --card-bg: rgba(25, 25, 25, 0.85);
    --card-border: rgba(255, 215, 0, 0.12);
}

/* App background */
.stApp {
    font-family: 'Poppins', sans-serif;
    background: radial-gradient(circle at top left, #111 0%, #000 30%, #0d0d0d 70%, #050505 100%);
    color: var(--text-primary);
}

/* App header centered & styled */
.app-header { text-align: center; margin: 2rem 0 1rem 0; }
.app-header .title-icon {
    font-size: 3.8rem;
    color: var(--primary-gold);
    text-shadow: 0 0 28px var(--gold-glow);
}
.app-header h1 {
    font-size: 2.8rem;
    margin: 0.3rem 0;
    font-weight: 700;
    background: linear-gradient(90deg, var(--primary-gold), var(--gold-hover));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Card style */
.card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.25rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

/* Gold buttons */
div.stButton > button, button[kind="primary"] {
    background: linear-gradient(135deg, #FFD700, #FFB700) !important;
    color: #000 !important;
    font-weight: 600;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    box-shadow: 0 6px 16px rgba(255,215,0,0.18) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #FFEA70, #FFD700) !important;
}

/* Section headings */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    text-align: center;
    margin-top: 1.25rem;
    margin-bottom: 0.6rem;
    background: linear-gradient(90deg, #FFD700, #FFEA70);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Result area */
.result-header { font-size: 1.05rem; color: var(--text-secondary); margin-bottom: 0.5rem; }
.suggestion-list li {
    border-left: 3px solid var(--primary-gold);
    background: rgba(255,255,255,0.03);
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 8px;
}

/* Disclaimer */
.disclaimer-box {
    text-align: center;
    color: var(--text-secondary);
    padding-top: 10px;
    font-size: 0.9rem;
}

/* Footer */
.footer { text-align: center; color: var(--text-secondary); margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--card-border); }
</style>
""", unsafe_allow_html=True)

# --- Symptom categories (kept from your original mapping) ---
symptom_categories = {
    "General & Systemic": ['itching', 'chills', 'fatigue', 'lethargy', 'malaise', 'weight_loss', 'weight_gain', 'excessive_hunger', 'dehydration', 'sweating', 'fever'],
    "Head & Neck": ['headache', 'dizziness', 'slurred_speech', 'sinus_pressure', 'runny_nose', 'congestion', 'sore_throat', 'stiff_neck', 'loss_of_smell', 'ulcers_on_tongue', 'patches_in_throat', 'enlarged_thyroid', 'puffy_face_and_eyes', 'swollen_lymph_nodes'],
    "Eyes & Vision": ['blurred_and_distorted_vision', 'yellowing_of_eyes', 'redness_of_eyes', 'pain_behind_the_eyes', 'sunken_eyes', 'visual_disturbances'],
    "Chest & Respiratory": ['chest_pain', 'breathlessness', 'cough', 'phlegm', 'mucoid_sputum', 'rusty_sputum', 'palpitations'],
    "Abdominal & Digestive": ['stomach_pain', 'acidity', 'vomiting', 'nausea', 'indigestion', 'diarrhoea', 'constipation', 'abdominal_pain', 'belly_pain', 'passage_of_gases', 'bloody_stool', 'stomach_bleeding', 'distention_of_abdomen'],
    "Skin & Joints": ['skin_rash', 'nodal_skin_eruptions', 'dischromic _patches', 'yellowish_skin', 'bruising', 'joint_pain', 'neck_pain', 'back_pain', 'knee_pain', 'hip_joint_pain', 'weakness_of_one_body_side', 'weakness_in_limbs', 'swelling_joints', 'movement_stiffness', 'swollen_legs', 'brittle_nails', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails'],
    "Urinary & Genital": ['burning_micturition', 'spotting_ urination', 'dark_urine', 'yellow_urine', 'abnormal_menstruation', 'continuous_feel_of_urine'],
    "Psychological & Mood": ['anxiety', 'mood_swings', 'depression', 'irritability', 'restlessness', 'lack_of_concentration', 'altered_sensorium', 'coma']
}

# --- Data Loading & Processing ---
@st.cache_data(show_spinner=False)
def load_data():
    """Loads model and CSVs; stops app with clear message on failure."""
    try:
        with open("disease_predictor.pkl", "rb") as f:
            model = pickle.load(f)
        medications_df = pd.read_csv("medications.csv")
        train_df = pd.read_csv("Training.csv").drop(columns=["Unnamed: 133"], errors='ignore')
        symptoms_list = sorted(train_df.drop("prognosis", axis=1).columns.tolist())
        return model, medications_df, symptoms_list
    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}. Please ensure model and CSVs are in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

model, medications_df, symptoms_list = load_data()

# --- Session State ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'history' not in st.session_state:
    st.session_state.history = []

# --- PDF Generation Class ---
class PDF(FPDF):
    def __init__(self, name="User", age="N/A"):
        super().__init__()
        self.user_name = name
        self.user_age = age
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        # Top title bar with gold line
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, f'DiagnoX AI Pro - Report', 0, 1, 'C')
        self.set_draw_color(255, 215, 0)
        self.set_line_width(0.9)
        self.line(10, 26, 200, 26)
        self.ln(4)

    def footer(self):
        self.set_y(-16)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'L')
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'R')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        # gold title
        self.set_text_color(184, 134, 11)
        self.cell(0, 8, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, body)
        self.ln(2)

    def add_diagnosis(self, diagnosis, probability):
        self.set_font('Helvetica', 'B', 11)
        self.cell(120, 8, f"{diagnosis}", 1, 0, 'L')
        self.set_font('Helvetica', '', 11)
        self.cell(60, 8, f"{probability*100:.2f}% Confidence", 1, 1, 'R')

# --- UI Rendering Functions ---
def render_header():
    st.markdown("""
    <div class='app-header'>
        <div class='title-icon'>🧬</div>
        <h1>DiagnoX AI Pro</h1>
        <p style="color: #bdbdbd; max-width:900px; margin: 0.35rem auto;">
            Your advanced health companion for differential diagnosis. Enter your details, select symptoms, and receive a personalized analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_marketing_sections():
    st.markdown("<div class='section-title'>About DiagnoX AI Pro</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#bdbdbd;'>An AI-driven assistant delivering fast, explainable differential diagnoses and actionable suggestions for users and clinicians.</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### ⚡ Fast & Accurate\nHigh-speed analysis trained on large datasets.")
    with cols[1]:
        st.markdown("### 🧠 Explainable\nTop predictions with reasoning & recommended next steps.")
    with cols[2]:
        st.markdown("### 🔒 Privacy-first\nLocal inference; we don't share your data.")

def render_input_form():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Symptom Analysis Engine", unsafe_allow_html=True)

    # User details
    col1, col2 = st.columns([2,1])
    with col1:
        user_name = st.text_input("Full name", placeholder="e.g., John Doe")
    with col2:
        user_age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1, format="%d")

    st.markdown("---")
    st.markdown("#### Select your symptoms")
    selected_symptoms = []
    # show categories and valid options
    for category, cat_symptoms in symptom_categories.items():
        valid_symptoms = [s for s in cat_symptoms if s in symptoms_list]
        if valid_symptoms:
            with st.expander(category):
                picks = st.multiselect(f"Select from {category}", options=valid_symptoms, label_visibility="collapsed", key=f"sel_{category}")
                selected_symptoms.extend(picks)

    st.markdown("---")
    severity = st.select_slider("Rate overall severity", options=['Mild', 'Moderate', 'Severe'], value='Moderate', key="severity_slider")

    st.write("")  # spacing
    analyze_pressed = st.button("Analyze Symptoms", use_container_width=True)

    # handle analyze click
    if analyze_pressed:
        if not user_name or user_age <= 0:
            st.warning("⚠ Please enter a valid name and age.")
        elif not selected_symptoms:
            st.warning("⚠ Please select at least one symptom for analysis.")
        else:
            with st.spinner("DiagnoX AI is analyzing..."):
                time.sleep(1.2)
                # create input vector
                input_data = [0] * len(symptoms_list)
                for s in selected_symptoms:
                    try:
                        idx = symptoms_list.index(s)
                        input_data[idx] = 1
                    except ValueError:
                        pass
                input_data = np.array(input_data).reshape(1, -1)

                try:
                    # predict
                    prediction_proba = model.predict_proba(input_data)[0]
                    top3_idx = np.argsort(prediction_proba)[-3:][::-1]
                    top_predictions = []
                    for i in top3_idx:
                        disease_name = str(model.classes_[i])
                        prob = float(prediction_proba[i])
                        # find suggestions
                        match = medications_df[medications_df["Disease"].str.lower().str.strip() == disease_name.lower().strip()]
                        suggestions = match["Suggestion"].tolist() if not match.empty else ["Consult a healthcare professional for accurate diagnosis."]
                        top_predictions.append({"disease": disease_name, "probability": prob, "suggestions": suggestions})

                    results = {
                        "user_name": user_name,
                        "user_age": user_age,
                        "selected_symptoms": selected_symptoms,
                        "severity": severity,
                        "top_predictions": top_predictions
                    }

                    st.session_state.analysis_results = results
                    # add to history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    top_disease = top_predictions[0]['disease'] if top_predictions else "N/A"
                    st.session_state.history.insert(0, f"{timestamp} - {top_disease}")
                    # navigate to results by rerunning so results block shows
                    st.rerun()

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

def render_results():
    results = st.session_state.analysis_results
    if not results:
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### Analysis for {results['user_name']}", unsafe_allow_html=True)

    if results['severity'] == "Severe":
        st.markdown("<div style='border:1px solid #ff4b4b;background:rgba(255,75,75,0.08);padding:10px;border-radius:8px;color:#ff4b4b;font-weight:700;'>❗ Symptoms marked SEVERE — seek immediate medical attention.</div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1,1.6])
    with left_col:
        st.markdown("<div class='result-header'>Your Inputs</div>", unsafe_allow_html=True)
        st.write(f"**Name:** {results['user_name']}")
        st.write(f"**Age:** {results['user_age']}")
        st.write(f"**Severity:** {results['severity']}")
        st.write("**Symptoms:**")
        st.info(", ".join([s.replace('_', ' ').title() for s in results['selected_symptoms']]))

    with right_col:
        st.markdown("<div class='result-header'>Top Differential Diagnoses</div>", unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            "Condition": [p['disease'] for p in results['top_predictions']],
            "Confidence": [p['probability'] for p in results['top_predictions']]
        })
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("Condition"))

    st.markdown("---")
    st.markdown("<div class='result-header'>Detailed Recommendations</div>", unsafe_allow_html=True)
    for i, pred in enumerate(results['top_predictions']):
        title = f"{i+1}. {pred['disease']} ({pred['probability']*100:.1f}% confidence)"
        with st.expander(title, expanded=(i == 0)):
            for s in pred['suggestions']:
                st.markdown(f"- {s}")

    # PDF generation & download
    pdf_bytes = create_pdf_report(results)
    filename = f"DiagnoX_Report_{results['user_name'].replace(' ','_')}.pdf"
    st.download_button("📥 Download Report as PDF", data=pdf_bytes, file_name=filename, mime="application/pdf", use_container_width=True)

    st.markdown("<div class='disclaimer-box'><strong>Disclaimer:</strong> This is an AI-generated insight and not a medical diagnosis. Consult a licensed medical professional for clinical decisions.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def create_pdf_report(results):
    pdf = PDF(name=results['user_name'], age=results['user_age'])
    pdf.add_page()

    # Patient summary
    pdf.chapter_title("Patient Input Summary")
    pdf.chapter_body(f"Name: {results['user_name']}\nAge: {results['user_age']}\nSymptom Severity: {results['severity']}\nSelected Symptoms: {', '.join([s.replace('_',' ').title() for s in results['selected_symptoms']])}")

    if results['severity'] == "Severe":
        pdf.set_text_color(255, 0, 0)
        pdf.chapter_body("⚠ WARNING: Symptoms were marked SEVERE. Seek immediate medical attention.")
        pdf.set_text_color(0, 0, 0)

    # Diagnosis
    pdf.chapter_title("Differential Diagnosis Results")
    for i, pred in enumerate(results['top_predictions']):
        pdf.add_diagnosis(f"{i+1}. {pred['disease']}", pred['probability'])
    pdf.ln(3)

    # Recommendations
    pdf.chapter_title("Detailed Recommendations")
    for i, pred in enumerate(results['top_predictions']):
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, f"{i+1}. {pred['disease']}", 0, 1, 'L')
        for suggestion in pred['suggestions']:
            pdf.set_font('Helvetica', '', 11)
            pdf.multi_cell(0, 6, f" - {suggestion}")
        pdf.ln(2)

    s = pdf.output(dest="S")
    pdf_bytes = s.encode("latin-1") if isinstance(s, str) else s
    return io.BytesIO(pdf_bytes).getvalue()

def render_footer():
    st.markdown("<div class='footer'>DiagnoX AI Pro &copy; 2025 | Advanced Insights by Vansh</div>", unsafe_allow_html=True)

# --- Main App Flow ---
if __name__ == "__main__":
    # Sidebar: history + controls
    with st.sidebar:
        st.markdown("<h3 style='text-align:left'>📜 Analysis Log</h3>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("Your session analyses will appear here.")
        else:
            for i, item in enumerate(st.session_state.history):
                st.success(item, icon="✅")
        if st.session_state.history and st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    # Page header
    render_header()

    # Marketing (top)
    render_marketing_sections()

    # Main area: input or results
    if st.session_state.analysis_results:
        render_results()
        if st.button("⬅ Start New Analysis"):
            st.session_state.analysis_results = None
            st.rerun()
    else:
        render_input_form()
        st.info("👆 Enter your details and select symptoms to analyze.", icon="ℹ️")

    # Footer
    render_footer()
