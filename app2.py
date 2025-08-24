import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time
from datetime import datetime
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(
    page_title="DiagnoX AI Pro | Differential Diagnosis Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Enhanced CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

:root {
    --primary-gold: #FFD700;
    --gold-hover: #FFEA70;
    --gold-glow: rgba(255, 215, 0, 0.4);
    --bg-dark-1: #050505;
    --bg-dark-2: #0d0d0d;
    --bg-dark-3: #1a1a1a;
    --text-primary: #f5f5f5;
    --text-secondary: #b5b5b5;
    --card-bg: rgba(25, 25, 25, 0.85);
    --card-border: rgba(255, 215, 0, 0.15);
    --font-family-main: 'Poppins', sans-serif;
    --font-family-mono: 'Roboto Mono', monospace;
    --danger-red: #ff4b4b;
}

/* 🌌 App Background */
.stApp {
    font-family: var(--font-family-main);
    background: radial-gradient(circle at top left, #111 0%, #000 30%, #0d0d0d 70%, #050505 100%);
    background-attachment: fixed;
    color: var(--text-primary);
    animation: bgPulse 12s infinite alternate;
}
@keyframes bgPulse {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

/* ✨ Cards */
.card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 22px;
    padding: 1.75rem;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    transition: all 0.35s ease;
}
.card:hover {
    border-color: var(--primary-gold);
    box-shadow: 0 0 35px var(--gold-glow);
    transform: translateY(-7px) scale(1.02);
}

/* 🌟 Header */
.app-header { text-align: center; margin-bottom: 3rem; }
.app-header .title-icon { 
    font-size: 4.2rem; 
    color: var(--primary-gold); 
    text-shadow: 0 0 40px var(--gold-glow); 
    animation: pulse-icon 2s infinite ease-in-out; 
}
@keyframes pulse-icon { 
    0%, 100% { transform: scale(1); filter: drop-shadow(0 0 8px var(--gold-glow)); } 
    50% { transform: scale(1.15); filter: drop-shadow(0 0 18px var(--gold-glow)); } 
}
.app-header h1 { 
    font-size: 3.4rem; 
    font-weight: 700; 
    margin-bottom: 0.5rem; 
    background: linear-gradient(90deg, var(--primary-gold), var(--gold-hover), var(--primary-gold)); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    background-size: 200% auto;
    animation: shineText 6s linear infinite;
}
@keyframes shineText {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.app-header p { font-size: 1.15rem; color: var(--text-secondary); max-width: 650px; margin: 0 auto; }

/* 📊 Result Section */
.result-container { padding: 2rem; margin-top: 1rem; }
.result-header { font-size: 1.2rem; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.7rem; }
.suggestion-list li { 
    border-left: 3px solid var(--primary-gold); 
    padding: 0.9rem 1.2rem; 
    border-radius: 12px; 
    margin-bottom: 0.85rem; 
    background: rgba(255, 255, 255, 0.06); 
    transition: all 0.25s ease; 
}
.suggestion-list li:hover { background: rgba(255, 215, 0, 0.08); box-shadow: inset 0 0 12px rgba(255, 215, 0, 0.15); }

.disclaimer-box { 
    font-size: 0.9rem; 
    color: var(--text-secondary); 
    text-align: center; 
    padding: 1rem; 
    border-top: 1px solid var(--card-border); 
    margin-top: 1.7rem; 
    background: rgba(15,15,15,0.7); 
    border-radius: 0 0 20px 20px; 
}

/* 🚨 Severity Warning */
.severity-warning {
    border: 1px solid var(--danger-red);
    background: rgba(255, 75, 75, 0.15);
    color: var(--danger-red);
    padding: 1rem;
    border-radius: 12px;
    margin-top: 1rem;
    text-align: center;
    font-weight: 700;
    box-shadow: 0 0 15px rgba(255, 75, 75, 0.3);
}

/* 📂 Expander */
.st-emotion-cache-116h4er, .st-emotion-cache-p5msec {
    border: 1px solid var(--card-border);
    border-radius: 16px;
    background-color: rgba(20,20,20,0.6);
    transition: border 0.3s ease;
}
.st-emotion-cache-116h4er:hover, .st-emotion-cache-p5msec:hover {
    border-color: var(--primary-gold);
    box-shadow: 0 0 12px var(--gold-glow);
}

/* ⚡ Footer */
.footer { 
    text-align: center; 
    padding: 2rem 0 1rem 0; 
    font-size: 0.95rem; 
    color: var(--text-secondary); 
    border-top: 1px solid var(--card-border); 
    margin-top: 3rem; 
    letter-spacing: 0.5px;
}
.footer:hover { color: var(--primary-gold); transition: color 0.3s ease; }
</style>
""", unsafe_allow_html=True)


# --- Data Loading & Processing ---
@st.cache_data
def load_data():
    """Loads all necessary data files with error handling."""
    try:
        with open("disease_predictor.pkl", "rb") as f:
            model = pickle.load(f)
        medications_df = pd.read_csv("medications.csv")
        train_df = pd.read_csv("Training.csv").drop(columns=["Unnamed: 133"], errors='ignore')
        symptoms_list = sorted(train_df.drop("prognosis", axis=1).columns.tolist())
        return model, medications_df, symptoms_list
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A required file was not found: {e.filename}. Please ensure 'disease_predictor.pkl', 'medications.csv', and 'Training.csv' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Fatal Error during data loading: {e}")
        st.stop()

model, medications_df, symptoms_list = load_data()

# Symptom Categorization for better UX
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

# --- Initialize Session State ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- PDF Generation Class ---
class PDF(FPDF):
    """PDF generation class for creating the final report."""
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'DiagnoX AI Pro - Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'R')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_diagnosis(self, diagnosis, probability):
        self.set_font('Arial', 'B', 11)
        self.cell(95, 8, f" {diagnosis}", 1, 0, 'L')
        self.set_font('Arial', '', 11)
        self.cell(95, 8, f"{probability*100:.2f}% Confidence", 1, 1, 'R')

# --- UI Rendering Functions ---
def render_header():
    """Renders the main header of the application."""
    st.markdown("""
        <div class='app-header'>
            <div class='title-icon'>🧬</div>
            <h1>DiagnoX AI Pro</h1>
            <p>Your advanced health companion for differential diagnosis. Select symptoms, specify severity, and receive a detailed analysis.</p>
        </div>
    """, unsafe_allow_html=True)

def render_input_form():
    """Renders the input form for symptoms and severity."""
    st.markdown("<br>", unsafe_allow_html=True)
    main_cols = st.columns([1, 1.5, 1])
    with main_cols[1]:
        with st.container():
            st.markdown("<div class='card input-card'>", unsafe_allow_html=True)
            st.markdown("<h2>Symptom Analysis Engine</h2>", unsafe_allow_html=True)

            selected_symptoms = []
            st.markdown("<h6>Select the symptoms you are experiencing:</h6>", unsafe_allow_html=True)
            for category, symptoms_in_category in symptom_categories.items():
                with st.expander(f"**{category}**"):
                    valid_symptoms = [s for s in symptoms_in_category if s in symptoms_list]
                    # THE FIX IS ON THE LINE BELOW: multilet -> multiselect
                    selections = st.multiselect(f"Select from {category}", options=valid_symptoms, label_visibility="collapsed")
                    selected_symptoms.extend(selections)
            
            st.markdown("<hr style='border-color: var(--card-border);'>", unsafe_allow_html=True)

            st.markdown("<h6>Rate the overall severity of your symptoms:</h6>", unsafe_allow_html=True)
            severity = st.select_slider(
                "Severity",
                options=['Mild', 'Moderate', 'Severe'],
                value='Moderate',
                label_visibility="collapsed"
            )

            st.write("") # Spacer
            if st.button("Analyze Symptoms", use_container_width=True):
                if not selected_symptoms:
                    st.warning("⚠️ Please select at least one symptom for analysis.")
                    st.session_state.analysis_results = None
                else:
                    with st.spinner(''):
                        st.markdown("""<div style="text-align:center; color:var(--primary-gold); font-family:var(--font-family-mono);">DIAGNOX AI IS ANALYZING...</div>""", unsafe_allow_html=True)
                        time.sleep(1.5)

                    input_data = [0] * len(symptoms_list)
                    for symptom in selected_symptoms:
                        if symptom in symptoms_list:
                            input_data[symptoms_list.index(symptom)] = 1
                    input_data = np.array(input_data).reshape(1, -1)

                    try:
                        prediction_proba = model.predict_proba(input_data)[0]
                        top3_indices = np.argsort(prediction_proba)[-3:][::-1]
                        
                        results = {
                            "selected_symptoms": selected_symptoms,
                            "severity": severity,
                            "top_predictions": []
                        }

                        for i in top3_indices:
                            disease_name = model.classes_[i]
                            probability = prediction_proba[i]
                            suggestion_row = medications_df[medications_df["Disease"].str.lower() == disease_name.lower()]
                            suggestions = suggestion_row["Suggestion"].tolist() if not suggestion_row.empty else ["Consult a healthcare professional for guidance."]
                            
                            results["top_predictions"].append({
                                "disease": disease_name,
                                "probability": probability,
                                "suggestions": suggestions
                            })
                        st.session_state.analysis_results = results
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
                        st.session_state.analysis_results = None
            st.markdown("</div>", unsafe_allow_html=True)

def render_results():
    """Renders the analysis results, chart, and download button."""
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.markdown("---")
        result_cols = st.columns([0.5, 2, 0.5])

        with result_cols[1]:
            st.markdown("<div class='card result-container'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>Analysis Results</h2>", unsafe_allow_html=True)
            
            if results['severity'] == 'Severe':
                st.markdown("<div class='severity-warning'>❗️ Your symptoms are marked as severe. This analysis is not a substitute for professional medical advice. Please seek immediate medical attention.</div>", unsafe_allow_html=True)

            res_layout = st.columns([1, 1.2])
            with res_layout[0]:
                st.markdown("<div class='result-header'>Your Inputs</div>", unsafe_allow_html=True)
                st.write(f"**Severity:** {results['severity']}")
                st.write("**Selected Symptoms:**")
                symptoms_str = ", ".join([s.replace('_', ' ').title() for s in results['selected_symptoms']])
                st.info(symptoms_str)
            
            with res_layout[1]:
                st.markdown("<div class='result-header'>Differential Diagnosis</div>", unsafe_allow_html=True)
                chart_data = pd.DataFrame({
                    "Condition": [p['disease'] for p in results['top_predictions']],
                    "Confidence": [p['probability'] for p in results['top_predictions']]
                })
                st.bar_chart(chart_data, x="Condition", y="Confidence")

            st.markdown("<hr style='border-color: var(--card-border); margin-top:1rem; margin-bottom:1rem;'>", unsafe_allow_html=True)
            
            st.markdown("<div class='result-header'>Detailed Breakdown & Recommendations</div>", unsafe_allow_html=True)
            for i, pred in enumerate(results['top_predictions']):
                expander_title = f"**{i+1}. {pred['disease']}** ({pred['probability']*100:.1f}% confidence)"
                with st.expander(expander_title, expanded=(i == 0)):
                    st.markdown("<ul class='suggestion-list'>", unsafe_allow_html=True)
                    for s in pred['suggestions']:
                        st.markdown(f"<li>{s}</li>", unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)
            
            st.write("")
            pdf_data = create_pdf_report(results)
            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_data,
                file_name=f"DiagnoX_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            st.markdown("<div class='disclaimer-box'><strong>Disclaimer:</strong> DiagnoX AI provides preliminary insights and is not a substitute for professional medical diagnosis. Consult a qualified doctor for accurate health advice.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

def create_pdf_report(results):
    """Generates a PDF report from the analysis results."""
    pdf = PDF()
    pdf.add_page()
    
    pdf.chapter_title("Patient Input Summary")
    pdf.chapter_body(
        f"Symptom Severity: {results['severity']}\n"
        f"Selected Symptoms: {', '.join([s.replace('_', ' ').title() for s in results['selected_symptoms']])}"
    )

    if results['severity'] == 'Severe':
        pdf.set_text_color(255, 0, 0)
        pdf.chapter_body(
            "⚠️ WARNING: Symptoms were marked as SEVERE. "
            "It is highly recommended to seek immediate medical attention from a healthcare professional."
        )
        pdf.set_text_color(0, 0, 0)
        
    pdf.chapter_title("Differential Diagnosis Results")
    for i, pred in enumerate(results['top_predictions']):
        pdf.add_diagnosis(f"{i+1}. {pred['disease']}", pred['probability'])
    pdf.ln(5)

    pdf.chapter_title("Detailed Recommendations")
    for i, pred in enumerate(results['top_predictions']):
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 10, f"{i+1}. {pred['disease']}", 0, 1, 'L')
        for suggestion in pred['suggestions']:
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 5, f" - {suggestion}")
        pdf.ln(3)

    # ✅ Always return bytes (safe for fpdf and fpdf2)
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):   # old fpdf returns str
        pdf_bytes = pdf_bytes.encode("latin-1")

    return pdf_bytes

    
def render_footer():
    """Renders the page footer."""
    st.markdown("<div class='footer'>DiagnoX AI Pro &copy; 2025 | Advanced Insights by Vansh</div>", unsafe_allow_html=True)

# --- Main App Flow ---
if __name__ == "__main__":
    render_header()
    render_input_form()

    if st.session_state.analysis_results:
        render_results()
    else:
        st.info("👆 Begin by selecting your symptoms and severity above, then click 'Analyze' for your differential diagnosis.")

    render_footer()

