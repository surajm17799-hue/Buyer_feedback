import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from io import BytesIO

# --- Page setup ---
st.set_page_config(page_title="Buyer Feedback Sentiment Analysis", layout="wide", initial_sidebar_state="expanded")

# --- CSS styling ---
st.markdown("""
<style>
.section-title {color: #1d4ed8; font-size: 20px; font-weight: 600; margin-top: 20px; margin-bottom: 15px;}
.dataframe tbody tr:nth-child(odd) {background-color: #e6f2ff !important;}
.dataframe tbody tr:nth-child(even) {background-color: #f2f8fc !important;}
.dataframe th {background-color: #cce5ff !important; color: black;}
div[role='radiogroup'] > label {
    border: 1px solid #ccc; border-radius: 6px;
    padding: 8px 16px; margin-right: 8px; margin-bottom: 8px;
    background-color: white; cursor: pointer;
}
div[role='radiogroup'] > label:hover {
    background-color: #f0f8ff; border-color: #1d4ed8;
}
div[role='radiogroup'] > label[aria-checked="true"] {
    background-color: #3b82f6 !important; color: white !important;
    border-color: #1e40af !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<style>
.css-18e3th9 { padding-top: 1rem; }
</style>
<div style="background: linear-gradient(99deg, #3b82f6, #3b82f6); 
            padding: 14px; border-radius: 8px; 
            color: #ffffff; text-align: center; margin-bottom: 20px;">
    <h1 style="font-size: 28px; margin-bottom: 8px;">📈 Buyer Feedback Sentiment Analysis</h1>
    <p style="font-size: 16px; opacity: 0.9;">Analyze customer sentiment from feedback data</p>
</div>
""", unsafe_allow_html=True)

# --- Google Drive Auth (Service Account) ---
@st.cache_resource
def authenticate_drive():
    SCOPES = [
        "https://www.googleapis.com/auth/drive.metadata.readonly",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["google_service_account"], scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

drive_service = authenticate_drive()

# Google Drive folder ID
FOLDER_ID = "1iskRT5FQjaFiRWu_qe6AzDQ1mlyYtC6n"

# --- List CSV files in folder ---
def list_csv_files(folder_id):
    query = f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return results.get("files", [])

file_list = list_csv_files(FOLDER_ID)
file_names = [f["name"] for f in file_list]

if not file_names:
    st.error("No CSV files found in the Google Drive folder.")
    st.stop()

# --- Feedback Source Selection ---
st.markdown("<div class='section-title'>📂 Select Feedback Source</div>", unsafe_allow_html=True)
feedback_sources = ["All", "Seller Relevance", "Play Store", "NPS", "App Internal"]
selected_source = st.radio("", feedback_sources, index=0, horizontal=True, label_visibility="collapsed")

# --- Map feedback source to CSV files (exclude QTR files) ---
source_file_map = {
    "Seller Relevance": [f for f in file_names if "seller" in f.lower() and "qtr" not in f.lower()],
    "Play Store": [f for f in file_names if "play" in f.lower() and "qtr" not in f.lower()],
    "NPS": [f for f in file_names if "nps" in f.lower() and "qtr" not in f.lower()],
    "App Internal": [f for f in file_names if "internal" in f.lower() and "qtr" not in f.lower()],
    "All": [f for f in file_names if "qtr" not in f.lower()]
}
filtered_files = source_file_map.get(selected_source, [])

if not filtered_files:
    st.error(f"No CSV files match the selected feedback source: {selected_source}")
    st.stop()

# --- Download CSV ---
def download_csv(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    file_data = request.execute()
    return file_data

# --- Load CSVs ---
dfs = []
for file_name in filtered_files:
    file_id = next(f["id"] for f in file_list if f["name"] == file_name)
    raw_data = download_csv(file_id)
    try:
        preview_df = pd.read_csv(BytesIO(raw_data), encoding="utf-8-sig", nrows=5)
        id_cols = [c for c in preview_df.columns if "id" in c.lower()]
        dtype_map = {col: str for col in id_cols}
        df_temp = pd.read_csv(BytesIO(raw_data), encoding="utf-8-sig", dtype=dtype_map if id_cols else None)
    except UnicodeDecodeError:
        preview_df = pd.read_csv(BytesIO(raw_data), encoding="ISO-8859-1", nrows=5)
        id_cols = [c for c in preview_df.columns if "id" in c.lower()]
        dtype_map = {col: str for col in id_cols}
        df_temp = pd.read_csv(BytesIO(raw_data), encoding="ISO-8859-1", dtype=dtype_map if id_cols else None)
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

if "rating" in df.columns:
    df["rating"] = df["rating"].astype("Int64")

# --- Detect Source Type ---
def detect_source(columns):
    cols = set(c.strip().lower() for c in columns)
    if {"source", "buyer", "comment", "reason"}.issubset(cols):
        return "nps_app_internal"
    play_store_required = {"id", "sentiment", "user", "reason", "reason2", "comment", "ratingmeaning", "rating", "reviewer_type"}
    if play_store_required.issubset(cols):
        return "play_store"
    elif {"fk_glusr_buyer_id", "iil_glusr_seller_id", "feedback_date"}.issubset(cols):
        return "seller_relevance"
    return "unknown"

source_type = detect_source(df.columns) if selected_source != "All" else "all"

# --- Category Selection ---
reason_col = None
if source_type in ["play_store", "seller_relevance"]:
    reason_col = next((col for col in df.columns if "reason2" in col.lower()), None)
    if not reason_col:
        reason_col = next((col for col in df.columns if "reason" in col.lower()), None)
else:
    reason_col = next((col for col in df.columns if "reason" in col.lower()), None)

if reason_col:
    category_counts = df[reason_col].dropna().value_counts(ascending=False)
    sorted_categories = category_counts.index.tolist()
    unique_categories = ["All"] + sorted_categories
    st.markdown("<div class='section-title'>📂 Select Feedback Categories</div>", unsafe_allow_html=True)

    if "show_more_cats" not in st.session_state:
        st.session_state.show_more_cats = False

    if not st.session_state.show_more_cats:
        display_categories = ["All"] + sorted_categories[:5]
        selected_categories = st.radio("", display_categories, index=0, horizontal=True, label_visibility="collapsed")
        if st.button("Show more categories"):
            st.session_state.show_more_cats = True
            st.rerun()
    else:
        display_categories = unique_categories
        selected_categories = st.radio("", display_categories, index=0, horizontal=True, label_visibility="collapsed")
        if st.button("Show less"):
            st.session_state.show_more_cats = False
            st.rerun()

    if selected_categories != "All":
        df = df[df[reason_col] == selected_categories]

# --- Quarterly Data Section ---
qtr_file_map = {
    "NPS": "NPS_QTR",
    "Seller Relevance": "Seller Relevance QTR",
    "App Internal": "App Internal QTR",
    "Play Store": "Play Store QTR"
}
if selected_source in qtr_file_map:
    qtr_file_keyword = qtr_file_map[selected_source].lower()
    qtr_file_name = next((f for f in file_names if qtr_file_keyword in f.lower()), None)
    if qtr_file_name:
        qtr_file_id = next(f["id"] for f in file_list if f["name"] == qtr_file_name)
        qtr_data = download_csv(qtr_file_id)
        try:
            df_qtr = pd.read_csv(BytesIO(qtr_data), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df_qtr = pd.read_csv(BytesIO(qtr_data), encoding="ISO-8859-1")

        styled_qtr = df_qtr.style.set_table_styles(
            [{'selector': 'th', 'props': [('background-color', '#cce5ff'), ('color', 'black')]}]
        )
        st.markdown(f"<div class='section-title'> {selected_source} Quarterly Data</div>", unsafe_allow_html=True)
        st.dataframe(styled_qtr, use_container_width=True)
    else:
        st.warning(f"{qtr_file_map[selected_source]} CSV not found in Google Drive folder.")

# --- Feedback Entries Table ---
st.markdown("<div class='section-title'>📋 Feedback Entries</div>", unsafe_allow_html=True)
st.write(f"**{len(df)} records found** (sample records)")

def highlight_negative(row):
    if "sentiment" in row.index and str(row["sentiment"]).strip().lower() == "negative":
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)

df_sample = df.head(100)
if "sentiment" in df_sample.columns:
    st.dataframe(df_sample.style.apply(highlight_negative, axis=1), use_container_width=True)
else:
    st.dataframe(df_sample, use_container_width=True)

# --- Buyer Verbatims Table ---
st.markdown("<div class='section-title'>🗣 Buyer Verbatims</div>", unsafe_allow_html=True)
comment_col = next((col for col in df.columns if "comment" in col.lower()), None)
if comment_col:
    st.dataframe(df[[comment_col]].dropna().drop_duplicates(), use_container_width=True)
else:
    st.warning("No 'comment' column found.")
