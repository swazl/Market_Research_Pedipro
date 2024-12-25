################################################################################
# ULTIMATE ADVANCED MARKET RESEARCH DASHBOARD
#
# Integrates:
# 1) File ingestion (CSV, XLSX, PDF, DOCX, images).
# 2) PDF text extraction & chart digitization.
# 3) T5 summarization, sentiment analysis, manual text entry.
# 4) SPSS-like stats with scikit-learn + statsmodels + pingouin.
# 5) More advanced table extraction with pdfplumber, plus option for Camelot/Tabula.
# 6) External data from APIs (Reddit, NewsAPI, Google, Yelp, etc.).
# 7) Potential advanced LLM usage (OpenAI, local LLaMA).
# 8) Workarounds for Torch '__path__._path' warning, blank screen issues, etc.
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import re
import tempfile
import cv2
import pytesseract
import pdfplumber
import docx2txt
import PyPDF2
from PIL import Image
from typing import List

# For chart digitization
import plotdigitizer

# Stats & ML
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import scipy.stats as stats      # For basic ANOVA
import statsmodels.api as sm     # For advanced regression
import pingouin as pg           # For advanced ANOVA, t-tests, etc.

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# NLP & Summaries
from textblob import TextBlob
import torch
from transformers import pipeline

# For external data requests
import requests

# If you want advanced PDF table extraction with Camelot:
# import camelot  # pip install camelot-py ghostscript
# If you want Tabula:
# import tabula   # pip install tabula-py

# If you want to use OpenAI or local LLaMA:
# import openai
# from llama_cpp import Llama

################################################################################
# Torch Summarizer: T5
################################################################################

try:
    summarizer = pipeline("summarization", model="t5-small", device=torch.device("cpu"))
except Exception:
    summarizer = None


################################################################################
# 1. HELPER FUNCTIONS: FILE READING
################################################################################

def read_pdf_text_via_pdfplumber(file_bytes_or_path):
    """Extract text from PDF using pdfplumber."""
    lines = []
    with pdfplumber.open(file_bytes_or_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                lines.append(txt)
    return "\n".join(lines)


def read_pdf_text_via_pypdf2(file_bytes_or_path):
    """Fallback PDF text extraction with PyPDF2."""
    try:
        pdf = PyPDF2.PdfReader(file_bytes_or_path)
        alltxt = []
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                alltxt.append(t)
        return "\n".join(alltxt)
    except:
        return ""


def read_docx_text(file_obj):
    """Extract text from Word doc using docx2txt."""
    try:
        with io.BytesIO(file_obj.read()) as temp_buffer:
            text = docx2txt.process(temp_buffer)
        return text if text else ""
    except:
        return ""


def read_csv_to_df(file_obj, skip_rows=0):
    """Read CSV into DataFrame, skipping user-specified rows."""
    return pd.read_csv(file_obj, skiprows=skip_rows)


def read_excel_to_df(file_obj, skip_rows=0):
    """Read XLSX into DataFrame, skipping user-specified rows."""
    import openpyxl
    return pd.read_excel(file_obj, skiprows=skip_rows)


def is_image_file(filename: str) -> bool:
    """Check if the file extension indicates an image."""
    ext = filename.lower().split(".")[-1]
    return ext in ["png", "jpg", "jpeg", "bmp", "tiff", "gif"]


################################################################################
# 2. CHART DETECTION & DIGITIZATION
################################################################################

def extract_images_from_pdf(pdf_path: str) -> List[Image.Image]:
    """Extract images from PDF via pdfplumber."""
    images = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for img_info in page.images:
                    image_data = page.extract_image(img_info["xref"])
                    if image_data:
                        pil_img = Image.open(io.BytesIO(image_data["image"]))
                        images.append(pil_img)
    except:
        pass
    return images


def detect_chart_in_image(pil_img: Image.Image) -> bool:
    """
    Heuristic chart detection:
    - OCR to see if 'axis', 'chart', etc. appear
    - Count digits
    - Edge detection
    """
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    ocr_result = pytesseract.image_to_string(cv_img)

    chart_kw = ["axis", "chart", "figure", "graph", "x-axis", "y-axis"]
    found_kw = any(k in ocr_result.lower() for k in chart_kw)
    numeric_count = len(re.findall(r"\d+", ocr_result))

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edgecount = np.count_nonzero(edges)

    if (found_kw or numeric_count > 5) and edgecount > 500:
        return True
    return False


def digitize_chart(pil_img: Image.Image, plot_type="scatter") -> pd.DataFrame:
    """
    Use plotdigitizer to automatically get (X, Y) from a chart image.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_img.save(tmp.name)
        tmp.flush()
        path = tmp.name

    df = pd.DataFrame()
    try:
        digitizer = plotdigitizer.Digitizer(
            image_path=path,
            plot_type=plot_type,
            xscale="linear",
            yscale="linear"
        )
        data = digitizer.get_data()
        df = pd.DataFrame(data, columns=["X", "Y"])
    except:
        pass

    try:
        os.remove(path)
    except:
        pass

    return df


################################################################################
# 3. SMART PARSING & AUTO ANALYSIS
################################################################################

def try_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Strip out non-numeric chars from object columns and cast to numeric."""
    new_df = df.copy()
    for col in new_df.columns:
        if new_df[col].dtype == object:
            new_df[col] = new_df[col].astype(str).str.replace(r"[^\d.\-+eE]", "", regex=True)
            try:
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
            except:
                pass
    return new_df


def parse_percentage(val: str):
    """Convert '29%', '0.29', '29.0' -> float(0.29). Return None if not parseable."""
    val = val.strip()
    if val.endswith('%'):
        try:
            num = float(val[:-1])
            return num / 100.0
        except:
            return None
    else:
        try:
            f = float(val)
            if f > 1.0:
                return f / 100.0
            else:
                return f
        except:
            return None


def parse_product_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example "auto analysis" that tries to find product -> usage rows.
    Returns a DataFrame with [Product, Usage].
    """
    pairs = []
    for _, row in df.iterrows():
        cleaned = [str(x).strip() for x in row if str(x).strip()]
        if len(cleaned) == 2:
            p, v = cleaned
            usage = parse_percentage(v)
            if usage is not None:
                pairs.append((p, usage))

    if pairs:
        return pd.DataFrame(pairs, columns=["Product", "Usage"])

    # fallback approach
    possible_rows = []
    for _, row in df.iterrows():
        cleaned = [str(x).strip() for x in row if str(x).strip()]
        if len(cleaned) >= 2:
            usage_val = parse_percentage(cleaned[-1])
            if usage_val is not None:
                product_str = " ".join(cleaned[:-1])
                possible_rows.append((product_str, usage_val))
    if possible_rows:
        return pd.DataFrame(possible_rows, columns=["Product", "Usage"])

    return pd.DataFrame()


################################################################################
# 4. TEXT ANALYSIS (SUMMARIES & SENTIMENT)
################################################################################

def local_summarize_text(
    text: str,
    max_length: int = 120,
    min_length: int = 40,
    bullet_points: bool = False
) -> str:
    """Summarize text with T5, chunked to 1000 chars."""
    if not summarizer:
        return "ERROR: T5 summarizer not loaded or installed."
    txt = text.strip()
    if not txt:
        return "No text to summarize."
    chunk_size = 1000
    pieces = []
    for i in range(0, len(txt), chunk_size):
        chunk_text = txt[i:i + chunk_size]
        out = summarizer(
            chunk_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        sum_text = out[0]["summary_text"]
        if bullet_points:
            sum_text = f"• {sum_text}"
        pieces.append(sum_text)
    return "\n".join(pieces)


def sentiment_of_text(text: str):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


################################################################################
# 5. PROOF-OF-CONCEPT: TEXT-BASED TABLE EXTRACTION
################################################################################

def extract_numeric_tables_from_text(text: str) -> pd.DataFrame:
    """
    Naive approach: each line -> parse floats -> if >=2 floats, consider a 'row'.
    Return stacked DataFrame. For real usage, try Camelot or Tabula for PDF tables.
    """
    lines = text.splitlines()
    rows = []
    for line in lines:
        tokens = line.split()
        numeric_tokens = []
        for t in tokens:
            # remove non-digit/decimal chars
            stripped = re.sub(r"[^\d.\-+eE]", "", t)
            try:
                val = float(stripped)
                numeric_tokens.append(val)
            except:
                pass
        if len(numeric_tokens) >= 2:
            rows.append(numeric_tokens)
    if rows:
        max_len = max(len(r) for r in rows)
        padded = [r + [np.nan]*(max_len - len(r)) for r in rows]
        df = pd.DataFrame(padded, columns=[f"Col_{i+1}" for i in range(max_len)])
        return df
    else:
        return pd.DataFrame()


################################################################################
# 6. EXTERNAL DATA FETCH (API EXAMPLES)
################################################################################

def fetch_reddit_data(subreddit="news", limit=10, client_id="", client_secret="", user_agent=""):
    """Simple example of grabbing top subreddit posts using the OAuth flow."""
    if not client_id or not client_secret or not user_agent:
        st.warning("Reddit credentials missing. Check your config.")
        return pd.DataFrame()
    base_url = "https://www.reddit.com"
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    headers = {"User-Agent": user_agent}
    data = {"grant_type": "client_credentials"}

    token_res = requests.post(f"{base_url}/api/v1/access_token", auth=auth, data=data, headers=headers)
    if token_res.status_code != 200:
        st.error("Failed to fetch Reddit token. Check credentials.")
        return pd.DataFrame()
    token = token_res.json().get("access_token")

    api_headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}
    url = f"{base_url}/r/{subreddit}/hot.json?limit={limit}"
    res = requests.get(url, headers=api_headers)
    if res.status_code != 200:
        st.error("Failed to fetch subreddit data.")
        return pd.DataFrame()
    posts = res.json().get("data", {}).get("children", [])
    data_rows = []
    for p in posts:
        post_data = p["data"]
        data_rows.append({
            "title": post_data.get("title",""),
            "score": post_data.get("score", 0),
            "author": post_data.get("author",""),
            "url": f'{base_url}{post_data.get("permalink","")}'
        })
    return pd.DataFrame(data_rows)


def fetch_newsapi_data(query="technology", api_key="", page_size=10):
    """NewsAPI example with a simple query."""
    if not api_key:
        st.warning("No NewsAPI key provided.")
        return pd.DataFrame()
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "relevancy"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        st.error("NewsAPI request failed.")
        return pd.DataFrame()
    articles = r.json().get("articles", [])
    rows = []
    for art in articles:
        rows.append({
            "title": art.get("title",""),
            "source": art.get("source",{}).get("name",""),
            "url": art.get("url",""),
            "publishedAt": art.get("publishedAt","")
        })
    return pd.DataFrame(rows)


################################################################################
# 7. MAIN STREAMLIT APP
################################################################################

def main():
    st.set_page_config(page_title="Ultimate Advanced Dashboard", layout="wide")

    # In-case Torch prints the weird '__path__._path' warning
    st.write("**Note**: If you see 'Tried to instantiate class __path__._path' in logs, "
             "try upgrading PyTorch or ignore if summarization still works.")

    st.title("Ultimate Advanced Market Research Dashboard")
    st.write("""
    Key Features:
    1. File upload (CSV, XLSX, PDF, DOCX, images) -> parse text, detect/ digitize charts.
    2. Summaries with T5, advanced SPSS-like stats with statsmodels + pingouin.
    3. Table extraction from text, or optional Camelot/Tabula for PDF tables.
    4. External data from Reddit, NewsAPI, etc.
    5. Potential advanced LLM usage (OpenAI, local LLaMA).
    """)

    # Session storage
    if "parsed_text" not in st.session_state:
        st.session_state["parsed_text"] = []
    if "uploaded_dfs" not in st.session_state:
        st.session_state["uploaded_dfs"] = []
    if "chart_dfs" not in st.session_state:
        st.session_state["chart_dfs"] = []
    if "api_credentials" not in st.session_state:
        st.session_state["api_credentials"] = {}

    tabs = st.tabs([
        "Upload & Parse",
        "Data Preview",
        "SPSS-Style & Advanced Stats",
        "Auto-Analysis",
        "LLM Summaries & Table Extraction",
        "External Data / APIs",
        "Deploy Info"
    ])

    ############################################################################
    # TAB 1: UPLOAD & PARSE
    ############################################################################
    with tabs[0]:
        st.header("Upload & Parse Files (XLSX, CSV, PDF, DOCX, images)")
        skip_rows = st.number_input("Skip top rows (Excel/CSV only)", 0, 50, 0)
        files = st.file_uploader("Drop your files here", accept_multiple_files=True)

        if files and st.button("Process Files"):
            st.session_state["parsed_text"] = []
            st.session_state["chart_dfs"] = []
            st.session_state["uploaded_dfs"] = []

            for f in files:
                fname = f.name
                st.subheader(f"Processing: " + fname)

                # PDF
                if fname.lower().endswith(".pdf"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                        tmp_pdf.write(f.read())
                        tmp_pdf.flush()
                        path_pdf = tmp_pdf.name

                    pdf_text = ""
                    try:
                        pdf_text = read_pdf_text_via_pdfplumber(path_pdf)
                    except:
                        pdf_text = read_pdf_text_via_pypdf2(path_pdf)

                    if pdf_text:
                        st.session_state["parsed_text"].append(pdf_text)
                        st.success(f"Extracted PDF text length={len(pdf_text)}.")

                    # extract images & digitize
                    st.write("Extracting images from PDF...")
                    images = extract_images_from_pdf(path_pdf)
                    if images:
                        st.info(f"Found {len(images)} images.")
                        for i, img in enumerate(images):
                            if detect_chart_in_image(img):
                                st.write(f"Chart-like image found (Image #{i+1}). Attempting data digitization...")
                                cdf = digitize_chart(img)
                                if not cdf.empty:
                                    cdf["Source"] = fname
                                    st.session_state["chart_dfs"].append(cdf)
                                    st.success(f"Extracted chart data: {len(cdf)} rows.")

                    try:
                        os.remove(path_pdf)
                    except:
                        pass

                # DOCX
                elif fname.lower().endswith(".docx") or fname.lower().endswith(".doc"):
                    doc_txt = read_docx_text(f)
                    if doc_txt:
                        st.session_state["parsed_text"].append(doc_txt)
                        st.success(f"Word text extracted: length={len(doc_txt)}.")

                # CSV
                elif fname.lower().endswith(".csv"):
                    cdf = read_csv_to_df(f, skip_rows=skip_rows)
                    cdf = try_cast_numeric(cdf)
                    st.session_state["uploaded_dfs"].append({"name": fname, "df": cdf})
                    st.write("Preview after numeric cast:")
                    st.dataframe(cdf.head())

                # XLSX
                elif fname.lower().endswith(".xlsx"):
                    xdf = read_excel_to_df(f, skip_rows=skip_rows)
                    xdf = try_cast_numeric(xdf)
                    st.session_state["uploaded_dfs"].append({"name": fname, "df": xdf})
                    st.write("Preview after numeric cast:")
                    st.dataframe(xdf.head())

                # IMAGE
                elif is_image_file(fname):
                    pil_img = Image.open(f)
                    st.image(pil_img, width=300, caption=f"Uploaded image: {fname}")
                    if detect_chart_in_image(pil_img):
                        chartdf = digitize_chart(pil_img)
                        if not chartdf.empty:
                            chartdf["Source"] = fname
                            st.session_state["chart_dfs"].append(chartdf)
                            st.success(f"Extracted {len(chartdf)} data points from chart.")
                else:
                    # Plain text fallback
                    try:
                        txt_data = f.read().decode("utf-8", errors="ignore")
                        if txt_data.strip():
                            st.session_state["parsed_text"].append(txt_data)
                            st.success("Interpreted file as plain text.")
                    except:
                        st.warning("Unsupported file or read error.")

            st.success("All files processed!")
            st.write("Now go to 'Data Preview' or 'SPSS-Style & Advanced Stats' or 'LLM Summaries' tabs.")

    ############################################################################
    # TAB 2: DATA PREVIEW
    ############################################################################
    with tabs[1]:
        st.header("Data Preview")
        # Show all DataFrames
        if st.session_state["uploaded_dfs"]:
            for dd in st.session_state["uploaded_dfs"]:
                st.subheader(dd["name"])
                st.dataframe(dd["df"].head(len(dd["df"])))
        else:
            st.info("No uploaded DataFrames to preview.")

        if st.session_state["chart_dfs"]:
            st.write("---")
            st.header("Digitized Chart DataFrames")
            for i, cdf in enumerate(st.session_state["chart_dfs"]):
                st.subheader(f"ChartDF #{i+1} (Source: {cdf.get('Source','unknown')})")
                st.dataframe(cdf.head(len(cdf)))

    ############################################################################
    # TAB 3: SPSS-Style & Advanced Stats (statsmodels/pingouin)
    ############################################################################
    with tabs[2]:
        st.header("SPSS-Style & Advanced Stats (Descriptives, Correlation, ANOVA, Regression)")

        df_list = [d["name"] for d in st.session_state["uploaded_dfs"]]
        chosen_df_name = st.selectbox("Select DataFrame", ["(None)"] + df_list)
        if chosen_df_name != "(None)":
            the_df = None
            for d in st.session_state["uploaded_dfs"]:
                if d["name"] == chosen_df_name:
                    the_df = d["df"]
                    break
            if the_df is not None and not the_df.empty:
                st.subheader("Descriptive Statistics")
                st.dataframe(the_df.describe(include="all"))

                st.subheader("Correlation Matrix (Numeric Only) + Heatmap")
                numeric_df = the_df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr = numeric_df.corr()
                    st.dataframe(corr)
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.columns,
                        colorscale="RdBu",
                        zmid=0
                    ))
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("No numeric columns for correlation.")

                # Pingouin Example: repeated-measures ANOVA or T-test
                st.subheader("Pingouin Example (Simple T-test, e.g. col1 vs col2)")

                numeric_cols = numeric_df.columns.tolist()
                if len(numeric_cols) >= 2:
                    colA = st.selectbox("Column A", numeric_cols)
                    colB = st.selectbox("Column B", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
                    if st.button("Run paired t-test"):
                        subd = the_df[[colA, colB]].dropna()
                        if not subd.empty:
                            result = pg.ttest(subd[colA], subd[colB], paired=True)
                            st.write(result)
                        else:
                            st.warning("No data after dropping NA rows.")
                else:
                    st.info("Need at least 2 numeric columns for T-test.")

                st.subheader("ANOVA (One-way with scipy.stats)")
                # We'll do one-way with group col
                cat_cols = [c for c in the_df.columns if c not in numeric_cols]
                if len(numeric_cols) >= 1 and len(cat_cols) >= 1:
                    anova_col = st.selectbox("Numeric column for ANOVA", numeric_cols)
                    group_col = st.selectbox("Grouping column (categorical)", cat_cols)
                    if st.button("Run ANOVA"):
                        subdf = the_df[[anova_col, group_col]].dropna()
                        groups_data = []
                        for cat_val, subgrp in subdf.groupby(group_col):
                            groups_data.append(subgrp[anova_col].values)
                        if len(groups_data) < 2:
                            st.warning("Need >=2 groups.")
                        else:
                            f_val, p_val = stats.f_oneway(*groups_data)
                            st.write(f"One-way ANOVA: F={f_val:.4f}, p={p_val:.3e}")

                st.subheader("Advanced Regression (statsmodels Multiple Regression)")
                if len(numeric_cols) > 1:
                    ycol = st.selectbox("Dependent (Y)", numeric_cols)
                    xcols = st.multiselect("Independent (X)", numeric_cols, default=[])
                    if ycol in xcols:
                        st.warning("Remove Y from X columns.")
                    elif xcols and st.button("Run statsmodels OLS"):
                        data = the_df[[ycol]+xcols].dropna()
                        if not data.empty:
                            Y = data[ycol]
                            X = data[xcols]
                            X = sm.add_constant(X)
                            model = sm.OLS(Y, X).fit()
                            st.text(model.summary())
                        else:
                            st.warning("No valid data for regression after dropping NAs.")
                else:
                    st.info("Need >=2 numeric columns for regression.")
            else:
                st.warning("That DataFrame is empty or not found.")
        else:
            st.info("Select a valid DataFrame to analyze.")

    ############################################################################
    # TAB 4: AUTO-ANALYSIS
    ############################################################################
    with tabs[3]:
        st.header("Auto-Analysis (Descriptive Stats, Correlation, Quick Chart)")

        df_list = [d["name"] for d in st.session_state["uploaded_dfs"]]
        chosen = st.selectbox("Pick DataFrame", ["(None)"] + df_list)
        if chosen != "(None)":
            chosen_df = None
            for d in st.session_state["uploaded_dfs"]:
                if d["name"] == chosen:
                    chosen_df = d["df"]
                    break
            if chosen_df is not None and not chosen_df.empty:
                st.write("**Descriptive Stats**")
                st.dataframe(chosen_df.describe(include="all"))

                numeric_df = chosen_df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.write("**Correlation Matrix**")
                    c = numeric_df.corr()
                    st.dataframe(c)
                    fig_h = go.Figure(data=go.Heatmap(
                        z=c.values,
                        x=c.columns, y=c.columns,
                        colorscale="RdBu", zmid=0
                    ))
                    st.plotly_chart(fig_h, use_container_width=True)

                    if len(numeric_df.columns) >= 2:
                        xcol = numeric_df.columns[0]
                        ycol = numeric_df.columns[1]
                        fig_sc = px.scatter(chosen_df, x=xcol, y=ycol, title="Auto Scatter Plot")
                        st.plotly_chart(fig_sc)
                        st.write(f"A quick chart of **{xcol}** vs **{ycol}**. "
                                 "Check correlation above for the numeric value.")
                else:
                    st.warning("No numeric columns for auto correlation or scatter.")
            else:
                st.warning("No data or empty DF.")
        else:
            st.info("Select a DataFrame for auto-analysis.")

        st.write("---")
        st.subheader("Auto Product vs. Percentage Detection")
        if df_list:
            dfname2 = st.selectbox("Select DF for product->% table", ["(None)"] + df_list)
            if dfname2 != "(None)":
                found_df = None
                for dd in st.session_state["uploaded_dfs"]:
                    if dd["name"] == dfname2:
                        found_df = dd["df"]
                        break
                if found_df is not None and not found_df.empty:
                    if st.button("Parse Product vs. %"):
                        resdf = parse_product_percent_table(found_df)
                        if resdf.empty:
                            st.warning("No product->% rows found.")
                        else:
                            st.success(f"Found {len(resdf)} rows.")
                            st.dataframe(resdf)
                            fig_bar = px.bar(resdf, x="Product", y="Usage", text="Usage")
                            fig_bar.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                            fig_bar.update_yaxes(tickformat=".0%")
                            st.plotly_chart(fig_bar)

    ############################################################################
    # TAB 5: LLM Summaries & Table Extraction
    ############################################################################
    with tabs[4]:
        st.header("LLM Summaries & Table Extraction from Text")

        all_text = "\n\n".join(st.session_state["parsed_text"])
        st.write(f"Total text chunks: {len(st.session_state['parsed_text'])}, combined length={len(all_text)} chars.")
        if all_text:
            bullet_style = st.checkbox("Bullet-point style summary?", value=False)
            max_len = st.slider("Max summary length per chunk", 50, 300, 120)
            min_len = st.slider("Min summary length per chunk", 10, max_len-10, 40)

            if st.button("Summarize Text with T5"):
                with st.spinner("Summarizing..."):
                    summary = local_summarize_text(
                        all_text,
                        max_length=max_len,
                        min_length=min_len,
                        bullet_points=bullet_style
                    )
                st.success("Done Summarizing!")
                st.write("### Summary:")
                st.write(summary)
                pol, subj = sentiment_of_text(all_text)
                st.write(f"**Sentiment**: Polarity={pol:.2f}, Subjectivity={subj:.2f}")

            st.write("---")
            st.subheader("Naive Numeric Table Extraction from Text")
            if st.button("Try Extracting Numeric Tables"):
                with st.spinner("Extracting numeric lines..."):
                    ndff = extract_numeric_tables_from_text(all_text)
                if ndff.empty:
                    st.warning("No numeric lines found.")
                else:
                    st.success("Found numeric rows from text:")
                    st.dataframe(ndff)
                    st.session_state["uploaded_dfs"].append({"name":"ExtractedFromText", "df":ndff})

            st.write("---")
            st.subheader("Optional: Advanced PDF Table Extraction (Camelot/Tabula) (Commented Code)")
            st.info("See code for commented sections on Camelot/Tabula usage. Uncomment to use if installed.")
        else:
            st.info("No text available. Upload PDF or doc or manual text first in 'Upload & Parse' tab.")

    ############################################################################
    # TAB 6: EXTERNAL DATA / APIs
    ############################################################################
    with tabs[5]:
        st.header("External Data & API Credentials")

        st.write("""
        **Potential APIs**:
        - Reddit, YouTube, NewsAPI, Google, Yelp, Stack Exchange, BLS, OpenAI GPT, etc.
        """)

        with st.expander("Enter / View / Update Your Credentials"):
            st.write("**Reddit**")
            reddit_id = st.text_input("Reddit Client ID", value=st.session_state["api_credentials"].get("REDDIT_CLIENT_ID",""))
            reddit_secret = st.text_input("Reddit Client Secret", value=st.session_state["api_credentials"].get("REDDIT_CLIENT_SECRET",""), type="password")
            reddit_ua = st.text_input("Reddit User-Agent", value=st.session_state["api_credentials"].get("REDDIT_USER_AGENT",""))

            st.write("**NewsAPI**")
            news_api_key = st.text_input("NewsAPI Key", value=st.session_state["api_credentials"].get("NEWS_API_KEY",""), type="password")

            st.write("**OpenAI** (for advanced GPT usage)")
            openai_key = st.text_input("OpenAI Key", value=st.session_state["api_credentials"].get("OPENAI_API_KEY",""), type="password")

            # Add more as needed: YouTube, Google, Yelp, StackEx, BLS, local LLaMA model path, etc.

            if st.button("Save Credentials"):
                st.session_state["api_credentials"]["REDDIT_CLIENT_ID"] = reddit_id
                st.session_state["api_credentials"]["REDDIT_CLIENT_SECRET"] = reddit_secret
                st.session_state["api_credentials"]["REDDIT_USER_AGENT"] = reddit_ua
                st.session_state["api_credentials"]["NEWS_API_KEY"] = news_api_key
                st.session_state["api_credentials"]["OPENAI_API_KEY"] = openai_key
                st.success("Credentials updated in session.")

        st.write("---")
        st.subheader("Fetch Data from Reddit or NewsAPI (Demo)")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Reddit**")
            sub = st.text_input("Subreddit", "news")
            limit = st.number_input("Posts limit", 1, 50, 10)
            if st.button("Fetch from Reddit"):
                creds = st.session_state["api_credentials"]
                df_red = fetch_reddit_data(
                    subreddit=sub,
                    limit=limit,
                    client_id=creds.get("REDDIT_CLIENT_ID",""),
                    client_secret=creds.get("REDDIT_CLIENT_SECRET",""),
                    user_agent=creds.get("REDDIT_USER_AGENT","")
                )
                if not df_red.empty:
                    st.dataframe(df_red)
                    st.session_state["uploaded_dfs"].append({"name":f"Reddit_{sub}", "df":df_red})

        with col2:
            st.write("**NewsAPI**")
            q = st.text_input("NewsAPI query", "technology")
            psize = st.number_input("Page size", 1, 100, 10)
            if st.button("Fetch from NewsAPI"):
                creds = st.session_state["api_credentials"]
                df_news = fetch_newsapi_data(
                    query=q,
                    api_key=creds.get("NEWS_API_KEY",""),
                    page_size=psize
                )
                if not df_news.empty:
                    st.dataframe(df_news)
                    st.session_state["uploaded_dfs"].append({"name":f"NewsAPI_{q}", "df":df_news})

        st.write("""
        Similarly, add code for Yelp, Google, Stack Exchange, BLS, or other data sources.
        """)

    ############################################################################
    # TAB 7: DEPLOY INFO
    ############################################################################
    with tabs[6]:
        st.header("Deployment Info & Torch '__path__._path' Fixes")
        st.markdown(r"""
**Local**:
```bash
pip install streamlit pandas numpy pdfplumber docx2txt PyPDF2 pillow opencv-python pytesseract plotdigitizer \
            torch transformers textblob scikit-learn plotly openpyxl requests scipy statsmodels pingouin pyarrow
streamlit run ultimate_advanced_dashboard.py""")

if __name__ == "__main__":
    main()

