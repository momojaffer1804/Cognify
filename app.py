import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
from io import BytesIO
from datetime import datetime
import io

# ===============================
# PDF Helper Configuration
# ===============================

binary_cols = [
    'MemoryComplaints',
    'BehavioralProblems',
    'Forgetfulness',
    'CardiovascularDisease',
    'Smoking'
]

pretty_names = {
    'MMSE': 'MMSE Score',
    'ADL': 'Activities of Daily Living',
    'MemoryComplaints': 'Memory Complaints',
    'BehavioralProblems': 'Behavioral Problems',
    'FunctionalAssessment': 'Functional Assessment',
    'Forgetfulness': 'Forgetfulness',
    'Age': 'Age',
    'CardiovascularDisease': 'Cardiovascular Disease',
    'Smoking': 'Smoking',
    'BMI': 'Body Mass Index'
}

def format_value(col, val):
    if col in binary_cols:
        return "Yes" if int(val) == 1 else "No"
    if isinstance(val, float):
        return round(val, 2)
    return str(val)

# ===============================
# PDF Generator Function
# ===============================

def generate_pdf(input_df, diagnosis, probability, risk_level):
    buffer = BytesIO()
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    W, H = A4
    c = canvas.Canvas(buffer, pagesize=A4)

    margin = 50
    box_w = W - 2 * margin

    

    # ── SECTION 1: HEADER ────────────────────────────────
    c.setFillColor(colors.HexColor('#1a1a2e'))
    c.rect(margin, H - 120, box_w, 60, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin + 16, H - 92, "Cognify")

    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor('#9ca3af'))
    c.drawString(margin + 16, H - 108, "Alzheimer's Risk Assessment Report")

    import random
    report_id = f"#{random.randint(10000, 99999)}"
    c.setFont("Helvetica", 9)
    c.drawRightString(W - margin - 16, H - 95, f"Report {report_id}")

    # ── DIVIDER LINE ─────────────────────────────────────
    def hline(y):
        c.setStrokeColor(colors.HexColor('#1a1a2e'))
        c.setLineWidth(1)
        c.line(margin, y, margin + box_w, y)

    # ── SECTION 2: DATE ──────────────────────────────────
    hline(H - 120)
    c.setFillColor(colors.HexColor('#f8f9fb'))
    c.rect(margin, H - 148, box_w, 28, fill=1, stroke=0)

    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor('#6b7280'))
    c.drawString(margin + 16, H - 139, f"Generated: {datetime.now().strftime('%d %B %Y')}")

    # ── SECTION 3: RESULT ────────────────────────────────
    hline(H - 148)
    result_bg = '#fff5f5' if diagnosis.startswith("Alzheimer") else '#f0fdf4'
    result_color = '#dc2626' if diagnosis.startswith("Alzheimer") else '#16a34a'
    result_text = f"HIGH RISK — {probability*100:.2f}%" if diagnosis.startswith("Alzheimer") else f"LOW RISK — {probability*100:.2f}%"

    c.setFillColor(colors.HexColor(result_bg))
    c.rect(margin, H - 192, box_w, 44, fill=1, stroke=0)

    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.HexColor(result_color))
    c.drawString(margin + 16, H - 168, "RESULT:")
    c.drawString(margin + 90, H - 168, f"[ {result_text} ]")

    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor('#6b7280'))
    c.drawString(margin + 16, H - 184, f"Risk Level: {risk_level}   |   Diagnosis: {diagnosis}")

    # ── SECTION 4: TABLE HEADER ──────────────────────────
    hline(H - 192)
    c.setFillColor(colors.HexColor('#f3f4f6'))
    c.rect(margin, H - 220, box_w, 28, fill=1, stroke=0)

    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.HexColor('#374151'))
    c.drawString(margin + 16, H - 210, "Feature")
    c.drawString(margin + 320, H - 210, "Value")

    # vertical divider in table
    def vline(x, y1, y2):
        c.setStrokeColor(colors.HexColor('#1a1a2e'))
        c.setLineWidth(0.5)
        c.line(x, y1, x, y2)

    hline(H - 220)

    # ── SECTION 5: TABLE ROWS ────────────────────────────
    pretty_names = {
        'MemoryComplaints': 'Memory Complaints',
        'BehavioralProblems': 'Behavioral Problems',
        'FunctionalAssessment': 'Functional Assessment',
        'MMSE': 'MMSE Score',
        'ADL': 'Activities of Daily Living',
        'Forgetfulness': 'Forgetfulness',
        'Age': 'Age',
        'CardiovascularDisease': 'Cardiovascular Disease',
        'Smoking': 'Smoking',
        'BMI': 'Body Mass Index (BMI)'
    }

    row_y = H - 220
    for i, col in enumerate(input_df.columns):
        row_h = 26
        bg = '#ffffff' if i % 2 == 0 else '#f8f9fb'
        c.setFillColor(colors.HexColor(bg))
        c.rect(margin, row_y - row_h, box_w, row_h, fill=1, stroke=0)

        val = input_df[col].values[0]
        if col in binary_cols:
            display = "Yes" if int(val) == 1 else "No"
        elif isinstance(val, float):
            display = str(round(val, 1))
        else:
            display = str(val)

        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor('#374151'))
        c.drawString(margin + 16, row_y - row_h + 8, pretty_names.get(col, col))

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin + 320, row_y - row_h + 8, display)

        hline(row_y - row_h)
        vline(margin + 300, row_y, row_y - row_h)

        row_y -= row_h

    # ── FOOTER ───────────────────────────────────────────
    c.setFillColor(colors.HexColor('#f8f9fb'))
    c.rect(margin, 60, box_w, 40, fill=1, stroke=0)

    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor('#9ca3af'))
    c.drawString(margin + 16, 88, "Cognify is not a substitute for professional medical diagnosis.")
    c.drawString(margin + 16, 74, "Consult a qualified neurologist for clinical assessment.")
    c.drawRightString(W - margin - 16, 80, "Page 1 of 1")
    # Draw outer border now that we know final row_y
    
    c.save()
    buffer.seek(0)
    return buffer


# ===============================
# Load Model
# ===============================

model = joblib.load('alzheimer_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Cognify", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Montserrat', sans-serif !important;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #f8f9fb;
    color: #1a1a2e;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
    max-width: 860px;
    margin: 0 auto;
    padding: 0 24px;
}
[data-testid="stAppViewContainer"] { background-color: #f8f9fb; }
[data-testid="stBottom"] { background-color: #f8f9fb; }

div.stSlider > label {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px;
}
div.stSelectbox > label {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px;
}

[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: #1a1a2e !important;
    border-color: #1a1a2e !important;
}

div.stButton > button {
    width: 100%;
    background: #1a1a2e;
    color: #ffffff;
    border: none;
    padding: 15px;
    border-radius: 8px;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-top: 24px;
}
div.stButton > button:hover {
    background: #2d2d4e;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(26,26,46,0.15);
}

div.stSpinner > div {
    border-top-color: #1a1a2e !important;
}
</style>
""", unsafe_allow_html=True)

# ── TOP BAR ──────────────────────────────────────────────
st.markdown("""
<div style="border-bottom: 1px solid #e5e7eb; padding: 48px 0 36px; margin-bottom: 48px; text-align:center;">
    <div style="font-size:5rem; font-weight:800; color:#1a1a2e; letter-spacing:-2px; margin-bottom:12px; line-height:1;">Cognify</div>
    <div style="font-size:0.7rem; color:#9ca3af; letter-spacing:4px; text-transform:uppercase;">Alzheimer's Early Risk Assessment</div>
</div>
""", unsafe_allow_html=True)

# ── SECTION LABEL ────────────────────────────────────────
def section_label(text):
    st.markdown(f'<div style="font-size:0.85rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#1a1a2e; margin-bottom:16px; margin-top:32px; border-left:3px solid #1a1a2e; padding-left:10px;">{text}</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<hr style="border:none; border-top:1px solid #e5e7eb; margin:32px 0;">', unsafe_allow_html=True)

# ── INPUTS ───────────────────────────────────────────────
section_label("Clinical Measurements")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 75)
    mmse = st.slider("MMSE (Mini-Mental State Examination) Score (0–30)", 0, 30, 15)
    functional = st.slider("Functional Assessment (0–10)", 0, 10, 5)
    adl = st.slider("ADL (Activities of Daily Living) Score (0–10)", 0, 10, 5)
    bmi = st.slider("BMI", 15.0, 40.0, 25.0)

with col2:
    memory = st.selectbox("Memory Complaints", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    behavioral = st.selectbox("Behavioral Problems", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    forgetfulness = st.selectbox("Forgetfulness", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cardiovascular = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

predict_btn = st.button("Run Analysis")

# ── PREDICTION ───────────────────────────────────────────
if predict_btn:

    input_data = pd.DataFrame([[
        memory, behavioral, functional, mmse, adl,
        forgetfulness, age, cardiovascular, smoking, bmi
    ]], columns=[
        'MemoryComplaints', 'BehavioralProblems', 'FunctionalAssessment',
        'MMSE', 'ADL', 'Forgetfulness', 'Age', 'CardiovascularDisease',
        'Smoking', 'BMI'
    ])

    with st.spinner("Analyzing patient data..."):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        risk_level = "High" if probability > 0.66 else "Moderate" if probability > 0.33 else "Low"

    divider()
    section_label("Diagnosis Result")

    # Result card
    if prediction == 1:
        st.markdown(f"""
        <div style="border:1px solid #fecaca; background:#fff5f5; border-radius:12px; padding:28px 32px; margin-bottom:20px;">
            <div style="font-size:0.62rem; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:#ef4444; margin-bottom:10px;">High Risk Detected</div>
            <div style="font-size:1.6rem; font-weight:800; color:#1a1a2e; margin-bottom:8px;">{probability*100:.1f}% Alzheimer's Risk</div>
            <div style="font-size:0.82rem; color:#6b7280; line-height:1.6;">Patient data indicates significant markers associated with Alzheimer's Disease. Further clinical evaluation is strongly recommended.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="border:1px solid #bbf7d0; background:#f0fdf4; border-radius:12px; padding:28px 32px; margin-bottom:20px;">
            <div style="font-size:0.62rem; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:#16a34a; margin-bottom:10px;">Low Risk</div>
            <div style="font-size:1.6rem; font-weight:800; color:#1a1a2e; margin-bottom:8px;">{probability*100:.1f}% Alzheimer's Risk</div>
            <div style="font-size:0.82rem; color:#6b7280; line-height:1.6;">No significant Alzheimer's markers detected based on current inputs. Continue routine monitoring.</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

    # Score boxes
    risk_color = "#ef4444" if probability > 0.66 else "#f59e0b" if probability > 0.33 else "#16a34a"

    st.markdown(f"""
    <div style="display:flex; gap:12px; margin-top:4px; margin-bottom:4px;">
        <div style="flex:1; background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:20px; text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:#1a1a2e;">{probability*100:.1f}%</div>
            <div style="font-size:0.62rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:#9ca3af; margin-top:4px;">Risk Score</div>
        </div>
        <div style="flex:1; background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:20px; text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:{risk_color};">{risk_level}</div>
            <div style="font-size:0.62rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:#9ca3af; margin-top:4px;">Risk Level</div>
        </div>
        <div style="flex:1; background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:20px; text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:#1a1a2e;">95%</div>
            <div style="font-size:0.62rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:#9ca3af; margin-top:4px;">Model Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk Meter
    divider()
    section_label("Risk Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={
            'suffix': '%',
            'font': {'size': 52, 'color': '#1a1a2e', 'family': 'Montserrat'},
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': '#9ca3af',
                'tickfont': {'color': '#6b7280', 'size': 12, 'family': 'Montserrat'}
            },
            'bar': {'color': "#1a1a2e", 'thickness': 0.03},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': '#22c55e'},
                {'range': [33, 66], 'color': '#f59e0b'},
                {'range': [66, 100], 'color': '#ef4444'}
            ],
            'threshold': {
                'line': {'color': '#1a1a2e', 'width': 6},
                'thickness': 0.85,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='#f8f9fb',
        plot_bgcolor='#f8f9fb',
        font_color='#1a1a2e',
        height=320,
        margin=dict(t=40, b=10, l=60, r=60)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── FEATURE CONTRIBUTION ─────────────────────────────
    divider()
    section_label("Feature Contribution Analysis")

    explainer = shap.TreeExplainer(model)
    input_data_scaled = pd.DataFrame(input_scaled, columns=input_data.columns)
    shap_values = explainer.shap_values(input_data_scaled)

    contrib_df = pd.DataFrame({
        'Feature': input_data.columns.tolist(),
        'Impact': shap_values[0]
    }).sort_values('Impact', ascending=False)

    contrib_df_sorted = contrib_df.sort_values('Impact', ascending=True)
    bar_colors = ['#ef4444' if v > 0 else '#16a34a' for v in contrib_df_sorted['Impact']]

    fig3 = go.Figure(go.Bar(
        x=contrib_df_sorted['Impact'],
        y=contrib_df_sorted['Feature'],
        orientation='h',
        marker_color=bar_colors,
        marker_line_width=0,
    ))

    fig3.update_layout(
        paper_bgcolor='#f8f9fb',
        plot_bgcolor='#f8f9fb',
        font=dict(family='Montserrat', color='#374151', size=12),
        height=360,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='#f3f4f6',
            zeroline=True,
            zerolinecolor='#e5e7eb',
            tickfont=dict(size=10, color='#9ca3af')
        ),
        yaxis=dict(
            tickfont=dict(size=11, color='#374151')
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div style="font-size:0.72rem; color:#9ca3af; margin-top:-8px;">
        <span style="color:#ef4444; font-weight:700;">■</span> Increases risk &nbsp;&nbsp;
        <span style="color:#16a34a; font-weight:700;">■</span> Decreases risk
    </div>
    """, unsafe_allow_html=True)

    # ── PDF DOWNLOAD ─────────────────────────────────────
    divider()
    section_label("Download Report")

    diagnosis = "Alzheimer's Disease Detected" if prediction == 1 else "No Alzheimer's Detected"
    pdf_buffer = generate_pdf(
        input_df=input_data,
        diagnosis=diagnosis,
        probability=probability,
        risk_level=risk_level
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="Cognify_report.pdf",
        mime="application/pdf"
    )

# ── DISCLAIMER ───────────────────────────────────────────
st.markdown("""
<div style="font-size:0.7rem; color:#d1d5db; text-align:center; padding:28px 0 8px; border-top:1px solid #e5e7eb; margin-top:48px; line-height:1.7;">
    Cognify is a research tool and does not constitute medical advice.<br>
    Always consult a qualified neurologist for clinical diagnosis.
</div>
""", unsafe_allow_html=True)