import streamlit as st
import pandas as pd
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

# --- Page setup ---
st.set_page_config(page_title="Buyer Feedback Sentiment Analysis", layout="wide", initial_sidebar_state="expanded")

# --- CSS styling ---
st.markdown("""
<style>
.section-title {color: #1d4ed8; font-size: 20px; font-weight: 600; margin-top: 20px; margin-bottom: 15px;}
.dataframe tbody tr:nth-child(odd) {background-color: #e6f2ff !important;}
.dataframe tbody tr:nth-child(even) {background-color: #f2f8fc !important;}
.dataframe th {background-color: #cce5ff !important; color: black;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div style="background: linear-gradient(99deg, #3b82f6, #3b82f6); 
            padding: 14px;
            border-radius: 8px; 
            color: #ffffff; 
            text-align: center; 
            margin-bottom: 20px;">
    <h1 style="font-size: 28px; margin-bottom: 8px;">📈 Buyer Feedback Sentiment Analysis</h1>
    <p style="font-size: 16px; opacity: 0.9;">Analyze customer sentiment from feedback data</p>
</div>
""", unsafe_allow_html=True)

# --- Google Drive Auth and file listing ---
def authenticate_drive():
    sa_info = dict(st.secrets["google_service_account"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp:
        json.dump(sa_info, temp)
        temp_path = temp.name

    gauth = GoogleAuth()
    gauth.LoadServiceConfigFile(temp_path)
    gauth.ServiceAuth()
    return GoogleDrive(gauth)

drive = authenticate_drive()


# Replace with your Google Drive folder ID here:
FOLDER_ID = "1iskRT5FQjaFiRWu_qe6AzDQ1mlyYtC6n"

# List CSV files in folder
file_list = drive.ListFile({'q': f"'{FOLDER_ID}' in parents and mimeType='text/csv' and trashed=false"}).GetList()
file_names = [f['title'] for f in file_list]

if not file_names:
    st.error("No CSV files found in the Google Drive folder.")
    st.stop()

st.markdown("<div class='section-title'>📂 Select Feedback Source </div>", unsafe_allow_html=True)
selected_file_name = st.selectbox("Choose a file", file_names)

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

def highlight_negative(row):
    if "sentiment" in row.index and str(row["sentiment"]).strip().lower() == "negative":
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)

if selected_file_name:
    # Get file ID and download the file content
    file_id = next(f['id'] for f in file_list if f['title'] == selected_file_name)
    file_obj = drive.CreateFile({'id': file_id})
    file_obj.GetContentFile(selected_file_name)

    try:
        df = pd.read_csv(selected_file_name, encoding="utf-8-sig", header=0)
    except UnicodeDecodeError:
        df = pd.read_csv(selected_file_name, encoding="ISO-8859-1", header=0)

    source_type = detect_source(df.columns)

    if source_type == "unknown":
        st.warning("Could not detect source type from CSV. Please check the column names.")
    else:
        st.success(f"Detected source type: **{source_type.replace('_', ' ').title()}**")

        # Show Buyer Verbatims
        st.markdown("<div class='section-title'>🗣 Buyer Verbatims</div>", unsafe_allow_html=True)
        comment_col = next((col for col in df.columns if "comment" in col.lower()), None)
        if comment_col:
            st.dataframe(df[[comment_col]].dropna().head(50), use_container_width=True)
        else:
            st.warning("No 'comment' column found.")

        # Select Category
        reason_col = None
        if source_type in ["play_store", "seller_relevance"]:
            reason_col = next((col for col in df.columns if "reason2" in col.lower()), None)
            if not reason_col:
                reason_col = next((col for col in df.columns if "reason" in col.lower()), None)
        else:
            reason_col = next((col for col in df.columns if "reason" in col.lower()), None)

        if reason_col:
            unique_categories = sorted(df[reason_col].dropna().unique())
            selected_categories = st.multiselect("Select Feedback Categories", unique_categories)

            if selected_categories:
                df_filtered = df[df[reason_col].isin(selected_categories)]
                st.markdown("<div class='section-title'>📋 Feedback Entries</div>", unsafe_allow_html=True)
                st.write(f"**{len(df_filtered)} records found**")

                if "sentiment" in df_filtered.columns:
                    st.dataframe(df_filtered.style.apply(highlight_negative, axis=1), use_container_width=True)
                else:
                    st.dataframe(df_filtered, use_container_width=True)
        else:
            st.warning("No 'reason' or 'reason2' column found.")
