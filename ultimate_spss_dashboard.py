###############################################################################
# ULTIMATE SPSS-STYLE MARKET RESEARCH DASHBOARD
#
# Key Additions in This Version:
# 1) SPSS-like Statistical Outputs (descriptive stats, correlation, ANOVA, regression).
# 2) Summaries & Auto-Analysis with textual explanation of charts.
# 3) Proof-of-concept table extraction from text.
# 4) Integration with multiple external APIs (Reddit, YouTube, NewsAPI, Google, Yelp,
#    Stack Exchange, BLS, OpenAI GPT, etc.) using your credentials.
# 5) Generalized for any industry: finance, economics, scientific research, etc.
###############################################################################

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

# For Chart digitization
import plotdigitizer

# ML and Stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import scipy.stats as stats  # for ANOVA, etc.

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# Text / NLP
from textblob import TextBlob

# Transformers Summarization (T5)
import torch
from transformers import pipeline

###############################################
# If needed on Windows, set Tesseract path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
###############################################

# Attempt to load T5 summarizer
try:
    summarizer = pipeline("summarization", model="t5-small", device=torch.device("cpu"))
except Exception:
    summarizer = None

################################################################################
# 1. HELPER FUNCTIONS: FILE READING
################################################################################

def read_pdf_text_via_pdfplumber(file_bytes_or_path):
    lines = []
    with pdfplumber.open(file_bytes_or_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                lines.append(txt)
    return "\n".join(lines)


def read_pdf_text_via_pypdf2(file_bytes_or_path):
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
    try:
        with io.BytesIO(file_obj.read()) as temp_buffer:
            text = docx2txt.process(temp_buffer)
        return text if text else ""
    except:
        return ""


def read_csv_to_df(file_obj, skip_rows=0):
    return pd.read_csv(file_obj, skiprows=skip_rows)


def read_excel_to_df(file_obj, skip_rows=0):
    import openpyxl
    return pd.read_excel(file_obj, skiprows=skip_rows)


def is_image_file(filename: str) -> bool:
    ext = filename.lower().split(".")[-1]
    return ext in ["png", "jpg", "jpeg", "bmp", "tiff", "gif"]


################################################################################
# 2. CHART DETECTION & DIGITIZATION
################################################################################

def extract_images_from_pdf(pdf_path: str) -> List[Image.Image]:
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
    """Heuristic approach for detecting chart-like images."""
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
    """Use plotdigitizer to attempt auto-digitizing charts."""
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
# 3. AUTO-ANALYSIS & SMART PARSING
################################################################################

def try_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to cast columns to numeric if possible by removing symbols."""
    new_df = df.copy()
    for col in new_df.columns:
        if new_df[col].dtype == object:
            # Remove non-numeric except digits, dot, sign, exponent
            new_df[col] = new_df[col].astype(str).str.replace(r"[^\d.\-+eE]", "", regex=True)
            try:
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
            except:
                pass
    return new_df


def parse_product_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple attempt to detect rows with a 'product' cell and a 'percentage' cell.
    Return DataFrame [Product, Usage].
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

    # Try fallback approach: last cell is usage
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


def parse_percentage(val: str):
    """Convert '29%', '0.29', '29.0' -> 0.29. Return None if not parseable."""
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


################################################################################
# 4. TEXT ANALYSIS
################################################################################

def local_summarize_text(
    text: str,
    max_length: int = 120,
    min_length: int = 40,
    bullet_points: bool = False
) -> str:
    """Summarize text with T5, chunked to 1000 chars. Return combined summary."""
    if not summarizer:
        return "ERROR: T5 summarizer not loaded or installed."
    text = text.strip()
    if not text:
        return "No text to summarize."
    chunk_size = 1000
    pieces = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size]
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
# 5. PROOF-OF-CONCEPT: EXTRACT TABLES FROM TEXT
################################################################################

def extract_numeric_tables_from_text(text: str) -> pd.DataFrame:
    """
    Very naive approach:
    - Looks for lines with numbers separated by spaces or tabs
    - Attempts to parse each line as columns of floats
    - Returns a "stacked" DataFrame with all numeric rows combined
    This is purely a proof-of-concept, not robust for real usage.
    """
    lines = text.splitlines()
    rows = []
    for line in lines:
        # Potentially skip lines that have too few digits
        tokens = line.split()
        numeric_tokens = []
        for t in tokens:
            # Try parse as float
            try:
                val = float(re.sub(r"[^\d.\-+eE]", "", t))
                numeric_tokens.append(val)
            except:
                pass
        # If we found 2 or more numeric tokens, consider it a row
        if len(numeric_tokens) >= 2:
            rows.append(numeric_tokens)
    if rows:
        max_len = max(len(r) for r in rows)
        # Pad shorter rows with NaN
        padded_rows = [r + [np.nan]*(max_len - len(r)) for r in rows]
        df = pd.DataFrame(padded_rows, columns=[f"Col_{i+1}" for i in range(max_len)])
        return df
    else:
        return pd.DataFrame()


################################################################################
# 6. API / EXTERNAL DATA FETCH (Reddit, NewsAPI, etc.)
################################################################################
import requests

def fetch_reddit_data(subreddit="news", limit=10,
                      client_id="", client_secret="", user_agent=""):
    """
    Simple example using Reddit credentials to fetch top posts from a subreddit.
    """
    if not client_id or not client_secret:
        st.warning("Reddit credentials not provided.")
        return pd.DataFrame()
    base_url = "https://www.reddit.com"
    headers = {"User-Agent": user_agent}
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    # Must request a token for script usage
    data = {"grant_type": "client_credentials"}
    token_res = requests.post(f"{base_url}/api/v1/access_token",
                              auth=auth, data=data, headers=headers)
    if token_res.status_code != 200:
        st.error("Failed to fetch Reddit token. Check credentials.")
        return pd.DataFrame()
    token = token_res.json()["access_token"]

    api_headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}
    url = f"{base_url}/r/{subreddit}/hot.json?limit={limit}"
    res = requests.get(url, headers=api_headers)
    if res.status_code != 200:
        st.error("Failed to fetch subreddit data.")
        return pd.DataFrame()
    posts = res.json().get("data", {}).get("children", [])
    data_rows = []
    for post in posts:
        p = post["data"]
        data_rows.append({
            "title": p.get("title", ""),
            "score": p.get("score", 0),
            "author": p.get("author", ""),
            "url": f'{base_url}{p.get("permalink","")}'
        })
    df = pd.DataFrame(data_rows)
    return df


def fetch_newsapi_data(query="technology",
                       api_key="", page_size=10):
    """
    Simple example with NewsAPI to fetch headlines.
    """
    if not api_key:
        st.warning("NewsAPI key not provided.")
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
    st.set_page_config(page_title="Ultimate SPSS-Style Dashboard", layout="wide")
    st.title("Ultimate SPSS-Style Market Research Dashboard")
    st.write("""
    **Key Features**:
    1. SPSS-like stats (descriptives, correlation, ANOVA, regression).
    2. T5 Summaries + attempt to extract numeric tables from text.
    3. External data from multiple APIs (Reddit, NewsAPI, etc.).
    4. Apply to any industry: finance, economics, scientific, etc.
    """)

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
        "SPSS-Style Analysis",
        "Auto-Analysis",
        "LLM Summaries & Table Extraction",
        "External Data",
        "Deploy Info"
    ])

    ###########################################################################
    # TAB 1: UPLOAD & PARSE
    ###########################################################################
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
                st.subheader(f"Processing: {fname}")

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

                    # Extract images -> detect chart
                    st.write("Extracting images from PDF...")
                    images = extract_images_from_pdf(path_pdf)
                    if images:
                        st.info(f"Found {len(images)} images.")
                        for i, img in enumerate(images):
                            if detect_chart_in_image(img):
                                st.write(f"Chart-like image found (Image #{i + 1}). Attempting data digitization...")
                                cdf = digitize_chart(img, plot_type="scatter")
                                if not cdf.empty:
                                    cdf["Source"] = fname
                                    st.session_state["chart_dfs"].append(cdf)
                                    st.success(f"Chart data: {len(cdf)} rows.")

                    # Remove temp file
                    try:
                        os.remove(path_pdf)
                    except:
                        pass

                # DOCX
                elif fname.lower().endswith(".docx") or fname.lower().endswith(".doc"):
                    doc_txt = read_docx_text(f)
                    if doc_txt:
                        st.session_state["parsed_text"].append(doc_txt)
                        st.success(f"Word text extracted, length={len(doc_txt)}.")

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
                        st.session_state["parsed_text"].append(txt_data)
                        st.success("Interpreted as plain text.")
                    except:
                        st.warning("Unsupported file or read error.")

            st.success("All files processed! Go to 'Data Preview' or 'SPSS-Style Analysis' or 'LLM Summaries' next.")

    ###########################################################################
    # TAB 2: DATA PREVIEW
    ###########################################################################
    with tabs[1]:
        st.header("Data Preview")
        if st.session_state["uploaded_dfs"]:
            for df_item in st.session_state["uploaded_dfs"]:
                st.subheader(df_item["name"])
                st.dataframe(df_item["df"].head(len(df_item["df"])))
        else:
            st.info("No DataFrames available.")

    ###########################################################################
    # TAB 3: SPSS-Style Analysis
    ###########################################################################
    with tabs[2]:
        st.header("SPSS-Style Analysis (EDA, Correlation, ANOVA, Regression)")

        # let user pick a DataFrame
        df_names = [d["name"] for d in st.session_state["uploaded_dfs"]]
        chosen_df_name = st.selectbox("Select DataFrame to Analyze", ["(None)"] + df_names)
        if chosen_df_name != "(None)":
            chosen_df = None
            for d in st.session_state["uploaded_dfs"]:
                if d["name"] == chosen_df_name:
                    chosen_df = d["df"]
                    break

            if chosen_df is not None and not chosen_df.empty:
                st.subheader("Descriptive Statistics")
                st.dataframe(chosen_df.describe(include='all'))

                st.subheader("Correlation Matrix & Heatmap (Numeric Columns Only)")
                numeric_df = chosen_df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr = numeric_df.corr()
                    st.dataframe(corr)
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.columns,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.warning("No numeric columns for correlation.")

                st.subheader("ANOVA (One-Way) Example")
                numeric_cols = [c for c in chosen_df.columns if pd.api.types.is_numeric_dtype(chosen_df[c])]
                if len(numeric_cols) >= 1:
                    anova_col = st.selectbox("Numeric column to test", numeric_cols)
                    group_col = st.selectbox("Group column (categorical)", [c for c in chosen_df.columns if c not in numeric_cols])
                    if st.button("Run One-Way ANOVA"):
                        # Drop NA
                        subdf = chosen_df[[anova_col, group_col]].dropna()
                        # We'll group by each category in group_col and gather as separate arrays
                        groups = []
                        for cat_value, group_df in subdf.groupby(group_col):
                            groups.append(group_df[anova_col].values)
                        if len(groups) > 1:
                            f_val, p_val = stats.f_oneway(*groups)
                            st.write(f"ANOVA result: F={f_val:.3f}, p={p_val:.3e}")
                        else:
                            st.warning("Need at least 2 groups for ANOVA.")
                else:
                    st.info("No numeric columns for ANOVA test.")

                st.subheader("Regression (Simple Linear) Example")
                if len(numeric_cols) >= 2:
                    xcol = st.selectbox("X column", numeric_cols)
                    ycol = st.selectbox("Y column", numeric_cols)
                    if st.button("Run Regression"):
                        # Drop NA
                        subdf = chosen_df[[xcol, ycol]].dropna()
                        if len(subdf) < 2:
                            st.warning("Not enough data after dropping NA.")
                        else:
                            X = subdf[xcol].values.reshape(-1,1)
                            Y = subdf[ycol].values
                            model = LinearRegression()
                            model.fit(X, Y)
                            slope = model.coef_[0]
                            intercept = model.intercept_
                            r_sq = model.score(X, Y)
                            # adjusted r_sq
                            n = len(X)
                            k = 1  # one predictor
                            adj_r_sq = 1 - (1-r_sq)*(n-1)/(n-k-1)
                            st.write(f"Slope={slope:.4f}, Intercept={intercept:.4f}")
                            st.write(f"R²={r_sq:.4f}, Adjusted R²={adj_r_sq:.4f}")

                            # Plot
                            fig_reg = px.scatter(subdf, x=xcol, y=ycol, title="Regression")
                            xline = np.linspace(subdf[xcol].min(), subdf[xcol].max(), 50)
                            yline = slope*xline + intercept
                            fig_reg.add_trace(go.Scatter(x=xline, y=yline, mode="lines", name="Fit"))
                            st.plotly_chart(fig_reg)
                else:
                    st.info("Need at least 2 numeric columns for regression.")
            else:
                st.warning("Chosen DataFrame is empty or not found.")
        else:
            st.info("Select a valid DataFrame to analyze.")

    ###########################################################################
    # TAB 4: Auto-Analysis
    ###########################################################################
    with tabs[3]:
        st.header("Auto-Analysis (Enhanced)")

        # Let user pick a DataFrame
        df_names = [d["name"] for d in st.session_state["uploaded_dfs"]]
        chosen_df_name = st.selectbox("Select DF for Auto-Analysis", ["(None)"] + df_names)
        if chosen_df_name != "(None)":
            chosen_df = None
            for d in st.session_state["uploaded_dfs"]:
                if d["name"] == chosen_df_name:
                    chosen_df = d["df"]
                    break
            if chosen_df is not None and not chosen_df.empty:
                st.write("**Descriptive Stats**")
                st.dataframe(chosen_df.describe(include='all'))
                # We can do correlation
                numeric_df = chosen_df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.write("**Correlation Matrix**")
                    corr = numeric_df.corr()
                    st.dataframe(corr)
                    # Heatmap
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.columns,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("No numeric columns for correlation.")

                # Create one "auto" chart
                # We pick the first 2 numeric columns for a quick scatter if available
                if len(numeric_df.columns) >= 2:
                    xcol = numeric_df.columns[0]
                    ycol = numeric_df.columns[1]
                    fig_scatter = px.scatter(chosen_df, x=xcol, y=ycol, title="Auto Scatter")
                    st.plotly_chart(fig_scatter)
                    # Provide a textual summary
                    st.write(f"**Auto Explanation**: This scatter plot shows the relationship between **{xcol}** and **{ycol}**. "
                             "Points trending upward might indicate a positive correlation. "
                             "Refer to the correlation matrix for exact correlation coefficient.")
                else:
                    st.warning("Not enough numeric columns to create an auto scatter plot.")

            else:
                st.warning("Chosen DF is empty.")
        else:
            st.info("Select a DataFrame for auto-analysis.")

    ###########################################################################
    # TAB 5: LLM Summaries & Table Extraction
    ###########################################################################
    with tabs[4]:
        st.header("LLM Summaries & Table Extraction from Text")

        all_text = "\n\n".join(st.session_state["parsed_text"])
        st.write(f"We have {len(st.session_state['parsed_text'])} text chunks. Combined length={len(all_text)} chars.")

        if all_text:
            bullet_style = st.checkbox("Bullet-point style summary?", value=False)
            max_len = st.slider("Max summary length", 50, 300, 120)
            min_len = st.slider("Min summary length", 10, max_len - 10, 40)

            if st.button("Summarize Text"):
                with st.spinner("Summarizing..."):
                    summary = local_summarize_text(all_text, max_length=max_len, min_length=min_len, bullet_points=bullet_style)
                st.success("Done Summarizing!")
                st.write("### Summary:")
                st.write(summary)

                pol, subj = sentiment_of_text(all_text)
                st.write(f"**Sentiment**: Polarity={pol:.2f}, Subjectivity={subj:.2f}")

            st.write("---")
            if st.button("Try Extracting Numeric Tables from Text"):
                with st.spinner("Extracting..."):
                    extracted_df = extract_numeric_tables_from_text(all_text)
                if extracted_df.empty:
                    st.warning("No numeric tables found from text.")
                else:
                    st.success("Found some numeric lines. Displaying DataFrame:")
                    st.dataframe(extracted_df.head(len(extracted_df)))
                    # Optionally store
                    st.session_state["uploaded_dfs"].append({"name": "ExtractedTableFromText", "df": extracted_df})
        else:
            st.info("No text found. Upload PDF/DOCX or plain text first.")

    ###########################################################################
    # TAB 6: EXTERNAL DATA (APIs)
    ###########################################################################
    with tabs[5]:
        st.header("External Data Sources & Credentials")

        # Show list of potential APIs for Market Research
        st.markdown("""
        ### Common APIs for Market Research
        - **Reddit** (public content, conversations, consumer sentiment)
        - **YouTube Data API** (video statistics, trending topics)
        - **NewsAPI** (latest news articles, public sentiment)
        - **Google Programmable Search** (custom search, global web results)
        - **Yelp** (consumer reviews on local businesses)
        - **Stack Exchange** (Q&A data in specialized fields)
        - **BLS** (Bureau of Labor Statistics, for labor/economic data)
        - **OpenAI GPT** (for advanced LLM completions or embeddings)
        - **Financial Data**: Yahoo Finance, Alpha Vantage, etc.
        """)

        with st.expander("Enter or View Your Saved Credentials"):
            # Let the user input each key/secret
            # e.g. st.text_input for each
            st.write("**Reddit**")
            reddit_id = st.text_input("Reddit Client ID", value=st.session_state["api_credentials"].get("REDDIT_CLIENT_ID",""))
            reddit_secret = st.text_input("Reddit Client Secret", value=st.session_state["api_credentials"].get("REDDIT_CLIENT_SECRET",""), type="password")
            reddit_ua = st.text_input("Reddit User-Agent", value=st.session_state["api_credentials"].get("REDDIT_USER_AGENT",""))

            st.write("**YouTube**")
            youtube_api_key = st.text_input("YouTube API Key", value=st.session_state["api_credentials"].get("YOUTUBE_API_KEY",""), type="password")

            st.write("**NewsAPI**")
            news_api_key = st.text_input("NewsAPI Key", value=st.session_state["api_credentials"].get("NEWS_API_KEY",""), type="password")

            st.write("**Google Programmable Search**")
            google_key = st.text_input("Google Search API Key", value=st.session_state["api_credentials"].get("GOOGLE_SEARCH_API_KEY",""), type="password")
            google_cx = st.text_input("Google Search Engine ID", value=st.session_state["api_credentials"].get("GOOGLE_SEARCH_ENGINE_ID",""))

            st.write("**Yelp**")
            yelp_key = st.text_input("Yelp API Key", value=st.session_state["api_credentials"].get("YELP_API_KEY",""), type="password")

            st.write("**Stack Exchange**")
            stack_id = st.text_input("StackEx Client ID", value=st.session_state["api_credentials"].get("STACKEXCHANGE_CLIENT_ID",""))
            stack_secret = st.text_input("StackEx Client Secret", value=st.session_state["api_credentials"].get("STACKEXCHANGE_CLIENT_SECRET",""), type="password")
            stack_key = st.text_input("StackEx Key", value=st.session_state["api_credentials"].get("STACKEXCHANGE_KEY",""), type="password")

            st.write("**BLS**")
            bls_key = st.text_input("BLS Key", value=st.session_state["api_credentials"].get("BLS_API_KEY",""), type="password")

            st.write("**OpenAI GPT**")
            openai_key = st.text_input("OpenAI API Key", value=st.session_state["api_credentials"].get("OPENAI_API_KEY",""), type="password")

            if st.button("Save All Credentials"):
                st.session_state["api_credentials"]["REDDIT_CLIENT_ID"] = reddit_id
                st.session_state["api_credentials"]["REDDIT_CLIENT_SECRET"] = reddit_secret
                st.session_state["api_credentials"]["REDDIT_USER_AGENT"] = reddit_ua
                st.session_state["api_credentials"]["YOUTUBE_API_KEY"] = youtube_api_key
                st.session_state["api_credentials"]["NEWS_API_KEY"] = news_api_key
                st.session_state["api_credentials"]["GOOGLE_SEARCH_API_KEY"] = google_key
                st.session_state["api_credentials"]["GOOGLE_SEARCH_ENGINE_ID"] = google_cx
                st.session_state["api_credentials"]["YELP_API_KEY"] = yelp_key
                st.session_state["api_credentials"]["STACKEXCHANGE_CLIENT_ID"] = stack_id
                st.session_state["api_credentials"]["STACKEXCHANGE_CLIENT_SECRET"] = stack_secret
                st.session_state["api_credentials"]["STACKEXCHANGE_KEY"] = stack_key
                st.session_state["api_credentials"]["BLS_API_KEY"] = bls_key
                st.session_state["api_credentials"]["OPENAI_API_KEY"] = openai_key
                st.success("Credentials updated!")

        st.write("---")
        st.subheader("Fetch Data from Reddit or NewsAPI (example)")

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Reddit Example")
            sub = st.text_input("Subreddit to fetch", "news")
            limit = st.number_input("Number of posts", 1, 50, 10)
            if st.button("Fetch from Reddit"):
                creds = st.session_state["api_credentials"]
                r_df = fetch_reddit_data(
                    subreddit=sub, limit=limit,
                    client_id=creds.get("REDDIT_CLIENT_ID"),
                    client_secret=creds.get("REDDIT_CLIENT_SECRET"),
                    user_agent=creds.get("REDDIT_USER_AGENT")
                )
                if not r_df.empty:
                    st.dataframe(r_df)
                    st.session_state["uploaded_dfs"].append({"name": f"Reddit_{sub}", "df": r_df})

        with col2:
            st.write("### NewsAPI Example")
            q = st.text_input("Query for NewsAPI", "technology")
            psize = st.number_input("Page size", 1, 100, 10)
            if st.button("Fetch from NewsAPI"):
                creds = st.session_state["api_credentials"]
                n_df = fetch_newsapi_data(query=q, api_key=creds.get("NEWS_API_KEY"), page_size=psize)
                if not n_df.empty:
                    st.dataframe(n_df)
                    st.session_state["uploaded_dfs"].append({"name": f"NewsAPI_{q}", "df": n_df})

        st.write("""
        *Similarly, you can add more code to fetch from YouTube, Google CSE, Yelp, StackEx, BLS, etc.
        The results can be appended as new DataFrames in your session.* 
        """)

    ###########################################################################
    # TAB 7: DEPLOY INFO
    ###########################################################################
    with tabs[6]:
        st.header("Deployment Info")
        st.markdown(r"""
1. **Local / PyCharm**  
   ```bash
   pip install streamlit pandas numpy pdfplumber docx2txt PyPDF2 pillow opencv-python pytesseract plotdigitizer \
               torch transformers textblob scikit-learn plotly openpyxl requests scipy pyarrow
   streamlit run ultimate_spss_dashboard.py""")
