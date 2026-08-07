import re
import pickle
import pandas as pd
import streamlit as st
from pypdf import PdfReader

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# MODEL LABELS
# =========================================================

LABELS_DISPLAY = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative"
}

LABEL_EMOJI = {
    "Positive": "😊",
    "Negative": "😞",
    "Neutral": "😐"
}

LABEL_COLOR = {
    "Positive": "#22c55e",
    "Negative": "#ef4444",
    "Neutral": "#a1a1aa"
}

# =========================================================
# CUSTOM CSS - COMPLETE DARK THEME
# =========================================================

st.markdown("""
<style>
    /* ---------- GLOBAL ---------- */
    .stApp {
        background: #08090d;
        color: #f5f5f5;
    }

    .main .block-container {
        max-width: 1050px;
        padding-top: 40px;
        padding-bottom: 60px;
    }

    /* ---------- REMOVE STREAMLIT UI ---------- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }

    /* ---------- TYPOGRAPHY ---------- */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    .stMarkdown p, .stMarkdown li, .stCaption {
        color: #a1a1aa !important;
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0d0f15;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #242631;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        color: #a1a1aa;
        font-weight: 600;
        padding: 0 24px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #c4b5fd;
    }

    .stTabs [aria-selected="true"] {
        background: #181a24 !important;
        color: #c4b5fd !important;
        border: 1px solid #3a3d4b;
    }

    /* ---------- TEXT AREA ---------- */
    .stTextArea textarea {
        background: #0b0d12 !important;
        color: #f4f4f5 !important;
        border: 1px solid #30333f !important;
        border-radius: 14px !important;
        padding: 18px !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }

    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 1px #8b5cf6 !important;
    }

    .stTextArea label {
        color: #a1a1aa !important;
    }

    /* ---------- BUTTON ---------- */
    .stButton > button {
        width: 100%;
        border-radius: 11px;
        min-height: 46px;
        border: 1px solid #6d5bd0;
        background: linear-gradient(135deg, #7c3aed, #6366f1);
        color: white;
        font-weight: 650;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #a78bfa;
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.25);
    }

    .stButton > button:active {
        transform: scale(0.98);
    }

    /* ---------- UPLOAD ---------- */
    [data-testid="stFileUploader"] {
        background: #0b0d12;
        border: 1px dashed #3a3d4b;
        border-radius: 16px;
        padding: 15px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] p {
        color: #a1a1aa !important;
    }

    /* ---------- RESULT CARD ---------- */
    .result-box {
        margin-top: 25px;
        padding: 30px;
        border-radius: 16px;
        background: #0b0d12;
        border: 1px solid #292c37;
        text-align: center;
    }

    .result-label {
        font-size: 32px;
        font-weight: 750;
        margin-top: 5px;
    }

    .result-confidence {
        color: #9295a0;
        margin-top: 8px;
        font-size: 14px;
    }

    /* ---------- METRICS ---------- */
    [data-testid="stMetric"] {
        background: #0c0e14;
        border: 1px solid #292c37;
        border-radius: 14px;
        padding: 18px;
        transition: all 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: #3a3d4b;
    }

    [data-testid="stMetricLabel"] {
        color: #8f929d !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ---------- DATAFRAME ---------- */
    [data-testid="stDataFrame"] {
        background: #0b0d12 !important;
        border: 1px solid #292c37 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    [data-testid="stDataFrame"] table {
        background: #0b0d12 !important;
    }

    [data-testid="stDataFrame"] th {
        background: #0d0f15 !important;
        color: #c4b5fd !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #292c37 !important;
    }

    [data-testid="stDataFrame"] td {
        color: #f4f4f5 !important;
        border-bottom: 1px solid #1a1c26 !important;
    }

    [data-testid="stDataFrame"] tr:hover {
        background: #0d0f15 !important;
    }

    /* ---------- SELECTBOX ---------- */
    [data-baseweb="select"] > div {
        background: #0b0d12 !important;
        border: 1px solid #30333f !important;
        border-radius: 10px !important;
        color: #f4f4f5 !important;
    }

    [data-baseweb="select"] > div:hover {
        border-color: #8b5cf6 !important;
    }

    [data-baseweb="select"] label {
        color: #a1a1aa !important;
    }

    [data-baseweb="select"] input {
        color: #f4f4f5 !important;
    }

    /* ---------- DOWNLOAD ---------- */
    .stDownloadButton > button {
        background: #0c0e14 !important;
        border: 1px solid #343744 !important;
        color: #f4f4f5 !important;
        border-radius: 11px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        border-color: #8b5cf6 !important;
        background: #12141c !important;
        color: #ffffff !important;
    }

    /* ---------- PROGRESS ---------- */
    .stProgress > div {
        background: #1a1c26 !important;
        border-radius: 20px !important;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #6366f1) !important;
        border-radius: 20px !important;
    }

    /* ---------- WARNING / ERROR ---------- */
    .stAlert {
        background: #0d0f15 !important;
        border: 1px solid #292c37 !important;
        border-radius: 12px !important;
        color: #f4f4f5 !important;
    }

    .stAlert svg {
        fill: #a1a1aa !important;
    }

    /* ---------- SPINNER ---------- */
    .stSpinner {
        color: #8b5cf6 !important;
    }

    /* ---------- DIVIDER ---------- */
    hr {
        border-color: #1a1c26 !important;
        margin: 1.5rem 0 !important;
    }

    /* ---------- CHART ---------- */
    [data-testid="stVegaLiteChart"] {
        background: #0b0d12 !important;
        border-radius: 14px !important;
        padding: 15px !important;
        border: 1px solid #292c37 !important;
    }

    /* ---------- MOBILE ---------- */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 30px 15px 60px;
        }

        [data-testid="stMetric"] {
            padding: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0 16px;
            font-size: 14px;
        }
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        return pickle.load(f)

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================================================
# CLASSIFY
# =========================================================

def classify(text):
    if not text or not text.strip():
        return "Neutral", 0.0

    model = load_model()
    cleaned = clean_text(text)

    if not cleaned:
        return "Neutral", 0.0

    prediction = model.predict([cleaned])[0]

    try:
        probabilities = model.predict_proba([cleaned])[0]
        confidence = float(max(probabilities))
    except AttributeError:
        confidence = 0.0

    label = LABELS_DISPLAY.get(prediction, prediction.title())
    return label, confidence

# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_feedbacks(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    feedbacks = [line.strip() for line in full_text.split("\n") if line.strip()]
    return feedbacks

# =========================================================
# HEADER
# =========================================================

st.title("💬 Sentiment Analysis")
st.markdown(
    "Analyze customer feedback using a machine learning model. "
    "Get instant **Positive**, **Negative**, or **Neutral** sentiment predictions."
)
st.markdown("---")

# =========================================================
# TABS
# =========================================================

tab_text, tab_pdf = st.tabs(["📝 Text Analysis", "📄 PDF Analysis"])

# =========================================================
# TEXT ANALYSIS
# =========================================================

with tab_text:
    st.subheader("Analyze Feedback")
    st.caption("Paste or type customer feedback below. The model will classify the sentiment.")

    text = st.text_area(
        "Feedback",
        placeholder="Paste your customer feedback here...",
        height=180,
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        analyze = st.button("Analyze Sentiment", type="primary")

    if analyze:
        if not text.strip():
            st.warning("Please enter some feedback first.")
        else:
            with st.spinner("Analyzing feedback..."):
                label, confidence = classify(text)

            color = LABEL_COLOR[label]
            emoji = LABEL_EMOJI[label]

            st.markdown(
                f"""
                <div class="result-box">
                    <div style="font-size:42px; margin-bottom:5px;">{emoji}</div>
                    <div class="result-label" style="color:{color};">{label}</div>
                    <div class="result-confidence">Model confidence: {confidence * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence)

# =========================================================
# PDF ANALYSIS
# =========================================================

with tab_pdf:
    st.subheader("Analyze PDF Feedback")
    st.caption("Upload a PDF containing multiple customer feedback responses. For best results, keep one feedback per line.")

    pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a text-based PDF containing customer feedback.",
        label_visibility="visible"
    )

    if pdf is not None:
        with st.spinner("Reading and analyzing PDF..."):
            feedbacks = extract_feedbacks(pdf)

        if not feedbacks:
            st.error("Could not extract text from this PDF. If it is a scanned PDF, OCR is required.")
        else:
            rows = []

            for feedback in feedbacks:
                label, confidence = classify(feedback)
                rows.append({
                    "Feedback": feedback,
                    "Sentiment": label,
                    "Confidence": f"{confidence * 100:.0f}%"
                })

            df = pd.DataFrame(rows)
            total = len(df)

            counts = df["Sentiment"].value_counts()
            positive = int(counts.get("Positive", 0))
            negative = int(counts.get("Negative", 0))
            neutral = int(counts.get("Neutral", 0))

            # Summary
            st.markdown("---")
            st.subheader(f"📊 {total} Feedbacks Analyzed")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("😊 Positive", positive, f"{positive / total * 100:.0f}%")

            with c2:
                st.metric("😞 Negative", negative, f"{negative / total * 100:.0f}%")

            with c3:
                st.metric("😐 Neutral", neutral, f"{neutral / total * 100:.0f}%")

            # Chart
            st.markdown("---")
            st.subheader("Sentiment Distribution")

            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [positive, neutral, negative]
            })

            st.bar_chart(chart_data.set_index("Sentiment"))

            # Results table
            st.markdown("---")
            st.subheader("Feedback Results")

            choice = st.selectbox(
                "Filter by sentiment",
                ["All", "Positive", "Negative", "Neutral"]
            )

            view = df if choice == "All" else df[df["Sentiment"] == choice]

            st.dataframe(view, use_container_width=True, hide_index=True)

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results as CSV",
                csv,
                "feedback_sentiment.csv",
                "text/csv"
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption("Logistic Regression + TF-IDF · Sentiment Analysis")