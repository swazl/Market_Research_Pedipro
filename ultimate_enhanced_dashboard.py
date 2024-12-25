################################################################################
# ULTIMATE ALL-INCLUSIVE MARKET RESEARCH DASHBOARD
#
# This code includes:
# 1) Multi-file ingest (PDF, XLSX, CSV, DOCX, images).
# 2) Chart digitization with plotdigitizer.
# 3) Camelot and tabula for advanced PDF table extraction (optional).
# 4) Local Summarization with T5 (transformers).
# 5) OpenAI GPT usage for advanced text or embeddings.
# 6) Local LLaMA usage via llama-cpp-python (optional).
# 7) SPSS-like stats with statsmodels, pingouin.
# 8) R and SAS bridging (rpy2 and saspy examples).
# 9) Vector-based "semantic search" (FAISS or similar).
# 10) External APIs (Reddit, NewsAPI, etc.).
# 11) Deployment instructions for Streamlit Cloud or Docker.
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

# Plot digitizer
import plotdigitizer

# Stats & ML
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import scipy.stats as stats
import statsmodels.api as sm
import pingouin as pg

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# NLP & Summaries
from textblob import TextBlob
import torch
from transformers import pipeline

# For advanced PDF table extraction
# (Camelot requires ghostscript installed system-wide)
try:
    import camelot  # pip install camelot-py ghostscript
    CAMELOT_AVAILABLE = True
except:
    CAMELOT_AVAILABLE = False

try:
    import tabula  # pip install tabula-py (requires Java)
    TABULA_AVAILABLE = True
except:
    TABULA_AVAILABLE = False

# If you want local LLaMA usage:
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except:
    LLAMA_AVAILABLE = False

# If you want R bridging:
try:
    import rpy2.robjects as ro
    R_AVAILABLE = True
except:
    R_AVAILABLE = False

# If you want SAS bridging:
try:
    import saspy
    SAS_AVAILABLE = True
except:
    SAS_AVAILABLE = False

# If you want embedding-based search with FAISS or similar
try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

# External data requests
import requests

# If you want OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False


################################################################################
# Summarizer: T5 local
################################################################################
try:
    summarizer = pipeline("summarization", model="t5-small", device=torch.device("cpu"))
    T5_AVAILABLE = True
except:
    summarizer = None
    T5_AVAILABLE = False


################################################################################
# 1. FILE READ HELPERS
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
# 3. SMART PARSING & AUTO-ANALYSIS
################################################################################

def try_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
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
    val = val.strip()
    if val.endswith('%'):
        try:
            num = float(val[:-1])
            return num/100.0
        except:
            return None
    else:
        try:
            f = float(val)
            if f>1.0:
                return f/100.0
            else:
                return f
        except:
            return None


def parse_product_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for _, row in df.iterrows():
        cleaned = [str(x).strip() for x in row if str(x).strip()]
        if len(cleaned) == 2:
            p, v = cleaned
            usage = parse_percentage(v)
            if usage is not None:
                pairs.append((p, usage))
    if pairs:
        return pd.DataFrame(pairs, columns=["Product","Usage"])

    possible_rows = []
    for _, row in df.iterrows():
        cleaned = [str(x).strip() for x in row if str(x).strip()]
        if len(cleaned) >=2:
            usage_val = parse_percentage(cleaned[-1])
            if usage_val is not None:
                product_str = " ".join(cleaned[:-1])
                possible_rows.append((product_str, usage_val))
    if possible_rows:
        return pd.DataFrame(possible_rows, columns=["Product","Usage"])

    return pd.DataFrame()


################################################################################
# 4. TEXT ANALYSIS (T5 Summaries, Sentiment, OpenAI, LLaMA)
################################################################################

def local_t5_summarize(text: str, max_length=120, min_length=40, bullet_points=False):
    if not T5_AVAILABLE or not summarizer:
        return "T5 summarizer not available."
    if not text.strip():
        return "No text to summarize."
    chunk_size = 1000
    out_pieces = []
    for i in range(0, len(text), chunk_size):
        c = text[i:i+chunk_size]
        result = summarizer(c, max_length=max_length, min_length=min_length, do_sample=False)
        s = result[0]["summary_text"]
        if bullet_points:
            s = f"• {s}"
        out_pieces.append(s)
    return "\n".join(out_pieces)


def text_sentiment(text: str):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def openai_summarize(text: str, openai_key: str, max_tokens=200):
    """
    Summarize with OpenAI GPT if you have an API key.
    """
    if not OPENAI_AVAILABLE:
        return "OpenAI library not installed."
    if not openai_key:
        return "No OpenAI key provided."
    import openai
    openai.api_key = openai_key
    prompt = f"Summarize the following:\n{text}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens
    )
    return response["choices"][0]["text"].strip()


def local_llama_summarize(text: str, model_path=""):
    """
    Summarize with local LLaMA model (llama-cpp-python).
    """
    if not LLAMA_AVAILABLE:
        return "llama-cpp-python not available or failed to build."
    if not model_path:
        return "No model path provided for LLaMA."
    llm = Llama(model_path=model_path)
    prompt = f"Summarize the following:\n{text}"
    output = llm(prompt, max_tokens=256)
    return output["choices"][0]["text"]


################################################################################
# 5. TABLE EXTRACTION (Camelot / tabula)
################################################################################

def extract_tables_camelot(pdf_path: str):
    """
    Use Camelot to extract tables from PDF if CAMELot is installed.
    Returns a list of DataFrames.
    """
    if not CAMELOT_AVAILABLE:
        return []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all")
        dfs = []
        for t in tables:
            dfs.append(t.df)
        return dfs
    except:
        return []


def extract_tables_tabula(pdf_path: str):
    """
    Use tabula to extract tables from PDF if Tabula is installed (requires Java).
    Returns a list of DataFrames.
    """
    if not TABULA_AVAILABLE:
        return []
    try:
        dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
        return dfs
    except:
        return []


################################################################################
# 6. R and SAS bridging (rpy2, saspy)
################################################################################

def run_r_code(r_code: str):
    if not R_AVAILABLE:
        return "rpy2 not available."
    import rpy2.robjects as ro
    return ro.r(r_code)


def run_sas_code(sas_code: str):
    if not SAS_AVAILABLE:
        return "saspy not available."
    # Must configure SAS session or local mode
    import saspy
    # For example, a local mode session might look like:
    # session = saspy.SASsession(cfgname='winlocal')
    # result = session.submit(sas_code)
    # return result
    return "SAS session code execution is environment-dependent. Example here only."


################################################################################
# 7. EMBEDDING-BASED SEARCH (FAISS) [OPTIONAL DEMO]
################################################################################

def build_faiss_index(text_list: List[str]):
    """
    This is a simple example that uses a huggingface embedding model
    to embed each text chunk. Then store in a FAISS index for similarity search.
    """
    if not FAISS_AVAILABLE:
        st.warning("FAISS not installed. Skipping embedding-based search.")
        return None, []
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_list, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def faiss_search(query: str, index, embeddings, text_list, top_k=3):
    """
    Perform a similarity search using FAISS on the stored embeddings.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        results.append({"text": text_list[idx], "distance": dist})
    return results


################################################################################
# 8. EXTERNAL DATA FETCH (Reddit, NewsAPI, etc.)
################################################################################

def fetch_reddit_data(subreddit="news", limit=10, client_id="", client_secret="", user_agent=""):
    if not client_id or not client_secret or not user_agent:
        st.warning("Reddit credentials missing.")
        return pd.DataFrame()
    import requests
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type":"client_credentials"}
    headers = {"User-Agent": user_agent}
    token_res = requests.post("https://www.reddit.com/api/v1/access_token",
                              auth=auth, data=data, headers=headers)
    if token_res.status_code != 200:
        st.error("Reddit token fetch failed.")
        return pd.DataFrame()
    token = token_res.json().get("access_token")

    api_headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    res = requests.get(url, headers=api_headers)
    if res.status_code != 200:
        st.error("Failed to fetch subreddit data.")
        return pd.DataFrame()
    posts = res.json().get("data", {}).get("children", [])
    data_rows = []
    for p in posts:
        pd_ = p["data"]
        data_rows.append({
            "title": pd_.get("title",""),
            "score": pd_.get("score",0),
            "author": pd_.get("author",""),
            "url": "https://www.reddit.com"+pd_.get("permalink","")
        })
    return pd.DataFrame(data_rows)


def fetch_newsapi_data(query="technology", api_key="", page_size=10):
    if not api_key:
        st.warning("No NewsAPI key.")
        return pd.DataFrame()
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "apiKey": api_key, "pageSize": page_size, "language":"en", "sortBy":"relevancy"}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        st.error("NewsAPI request failed.")
        return pd.DataFrame()
    articles = r.json().get("articles",[])
    rows = []
    for a in articles:
        rows.append({
            "title": a.get("title",""),
            "source": a.get("source",{}).get("name",""),
            "url": a.get("url",""),
            "publishedAt": a.get("publishedAt","")
        })
    return pd.DataFrame(rows)


################################################################################
# MAIN STREAMLIT APP
################################################################################

def main():
    st.set_page_config(page_title="Ultimate All-Inclusive Dashboard", layout="wide")
    st.write("""
    # Ultimate All-Inclusive Market Research Dashboard

    This includes everything: 
    - PDF ingest (including Camelot/Tabula)
    - T5 & OpenAI Summaries
    - Local LLaMA
    - R & SAS bridging
    - SPSS-like stats (statsmodels, pingouin)
    - Embedding-based search with FAISS
    - External data from Reddit, NewsAPI, etc.
    - Single or multi-file ingestion for an Executive Report

    **Note**: Not all optional features are guaranteed to work out-of-the-box 
    unless you install and configure them properly. 
    But this code demonstrates the "big picture." 
    """)

    if "parsed_text" not in st.session_state:
        st.session_state["parsed_text"] = []
    if "uploaded_dfs" not in st.session_state:
        st.session_state["uploaded_dfs"] = []
    if "chart_dfs" not in st.session_state:
        st.session_state["chart_dfs"] = []
    if "api_creds" not in st.session_state:
        st.session_state["api_creds"] = {}
    if "faiss_index" not in st.session_state:
        st.session_state["faiss_index"] = None
    if "text_list" not in st.session_state:
        st.session_state["text_list"] = []
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"] = None

    tabs = st.tabs([
        "Upload & Parse",
        "Data Preview",
        "SPSS-Like & Advanced Stats",
        "Auto-Analysis",
        "LLM Summaries",
        "Table Extraction (Camelot/Tabula)",
        "R & SAS",
        "Embedding Search",
        "External Data",
        "Deploy Info"
    ])

    ########################################################################
    # TAB 1: UPLOAD & PARSE
    ########################################################################
    with tabs[0]:
        st.header("Upload & Parse Multiple Documents")
        skip_rows = st.number_input("Skip rows (Excel/CSV)",0,50,0)
        files = st.file_uploader("Drop your files here", accept_multiple_files=True)

        if files and st.button("Process Files"):
            st.session_state["parsed_text"] = []
            st.session_state["uploaded_dfs"] = []
            st.session_state["chart_dfs"] = []

            for f in files:
                fname = f.name
                st.subheader(f"Processing {fname}")

                if fname.lower().endswith(".pdf"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                        tmp_pdf.write(f.read())
                        tmp_pdf.flush()
                        path_pdf = tmp_pdf.name
                    pdf_txt = ""
                    try:
                        pdf_txt = read_pdf_text_via_pdfplumber(path_pdf)
                    except:
                        pdf_txt = read_pdf_text_via_pypdf2(path_pdf)

                    if pdf_txt.strip():
                        st.session_state["parsed_text"].append(pdf_txt)
                        st.success(f"Extracted PDF text length={len(pdf_txt)}.")

                    images = extract_images_from_pdf(path_pdf)
                    if images:
                        st.write(f"Found {len(images)} images, checking charts...")
                        for i, img in enumerate(images):
                            if detect_chart_in_image(img):
                                cdf = digitize_chart(img)
                                if not cdf.empty:
                                    cdf["Source"] = fname
                                    st.session_state["chart_dfs"].append(cdf)
                                    st.success(f"Digitized chart: {len(cdf)} rows.")
                    try:
                        os.remove(path_pdf)
                    except:
                        pass

                elif fname.lower().endswith(".doc") or fname.lower().endswith(".docx"):
                    txt = read_docx_text(f)
                    if txt.strip():
                        st.session_state["parsed_text"].append(txt)
                        st.success(f"Extracted {len(txt)} doc text.")
                elif fname.lower().endswith(".csv"):
                    cdf = read_csv_to_df(f, skip_rows=skip_rows)
                    cdf = try_cast_numeric(cdf)
                    st.session_state["uploaded_dfs"].append({"name":fname, "df":cdf})
                    st.dataframe(cdf.head())
                elif fname.lower().endswith(".xlsx"):
                    xdf = read_excel_to_df(f, skip_rows=skip_rows)
                    xdf = try_cast_numeric(xdf)
                    st.session_state["uploaded_dfs"].append({"name":fname, "df":xdf})
                    st.dataframe(xdf.head())
                elif is_image_file(fname):
                    pil_img = Image.open(f)
                    st.image(pil_img, width=300)
                    if detect_chart_in_image(pil_img):
                        cdf = digitize_chart(pil_img)
                        if not cdf.empty:
                            cdf["Source"] = fname
                            st.session_state["chart_dfs"].append(cdf)
                            st.success(f"Digitized chart data: {len(cdf)} rows.")
                else:
                    # fallback: plain text
                    txt_data = f.read().decode("utf-8", errors="ignore")
                    if txt_data.strip():
                        st.session_state["parsed_text"].append(txt_data)
                        st.success("Interpreted file as plain text.")

            st.success("All Files Processed!")

    ########################################################################
    # TAB 2: DATA PREVIEW
    ########################################################################
    with tabs[1]:
        st.header("Preview Ingested Data & Text")
        # show dataframes
        for d in st.session_state["uploaded_dfs"]:
            st.subheader(d["name"])
            st.dataframe(d["df"].head(len(d["df"])))
        # show chart dfs
        if st.session_state["chart_dfs"]:
            st.write("---")
            st.header("Chart-Based DFs")
            for i, cdf in enumerate(st.session_state["chart_dfs"]):
                st.subheader(f"Chart DF #{i+1}")
                st.dataframe(cdf)
        # show text
        if st.session_state["parsed_text"]:
            st.write("---")
            st.header("Text Documents")
            for i, t in enumerate(st.session_state["parsed_text"]):
                st.subheader(f"Text Chunk #{i+1} (length={len(t)})")
                st.text(t[:500] + "...")  # show snippet

    ########################################################################
    # TAB 3: SPSS-Like & Advanced Stats
    ########################################################################
    with tabs[2]:
        st.header("SPSS-like Stats: correlation, regression, anova, etc.")
        df_names = [d["name"] for d in st.session_state["uploaded_dfs"]]
        choice = st.selectbox("Select DF for analysis", ["(None)"]+df_names)
        if choice!="(None)":
            chosen_df = None
            for dd in st.session_state["uploaded_dfs"]:
                if dd["name"]==choice:
                    chosen_df = dd["df"]
                    break
            if chosen_df is not None and not chosen_df.empty:
                st.write("**Descriptive**")
                st.dataframe(chosen_df.describe(include="all"))

                numeric_df = chosen_df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.write("**Correlation**")
                    c = numeric_df.corr()
                    st.dataframe(c)
                    fig_h = go.Figure(data=go.Heatmap(
                        z=c.values, x=c.columns, y=c.columns, colorscale="RdBu", zmid=0
                    ))
                    st.plotly_chart(fig_h, use_container_width=True)

                    st.subheader("One-Way ANOVA (scipy.stats)")
                    cat_cols = [c for c in chosen_df.columns if c not in numeric_df.columns]
                    if cat_cols and numeric_df.columns.size>=1:
                        anova_col = st.selectbox("Numeric col for ANOVA", numeric_df.columns)
                        group_col = st.selectbox("Group col", cat_cols)
                        if st.button("Run ANOVA"):
                            sub = chosen_df[[anova_col, group_col]].dropna()
                            groups_data = []
                            for val, gp in sub.groupby(group_col):
                                groups_data.append(gp[anova_col].values)
                            if len(groups_data)>=2:
                                f_val, p_val = stats.f_oneway(*groups_data)
                                st.write(f"F={f_val:.4f}, p={p_val:.4e}")
                            else:
                                st.warning("Need >=2 groups.")

                    st.subheader("Multiple Regression (statsmodels)")
                    ncols = numeric_df.columns.tolist()
                    if len(ncols)>1:
                        ycol = st.selectbox("Y col", ncols)
                        xcols = st.multiselect("X cols", ncols, default=[ncols[0]] if ncols[0]!=ycol else [])
                        if ycol in xcols:
                            st.warning("Remove Y from X.")
                        elif xcols and st.button("Run Regression"):
                            data = chosen_df[[ycol]+xcols].dropna()
                            if not data.empty:
                                Y = data[ycol]
                                X = sm.add_constant(data[xcols])
                                model = sm.OLS(Y, X).fit()
                                st.text(model.summary())
                else:
                    st.warning("No numeric columns for correlation or regression.")
            else:
                st.warning("That DF is empty.")
        else:
            st.info("Select a DataFrame.")

        st.write("---")
        st.subheader("pingouin Example (Paired T-Test)")
        if choice!="(None)" and chosen_df is not None:
            numeric_cols = chosen_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols)>=2:
                colA = st.selectbox("Col A", numeric_cols)
                colB = st.selectbox("Col B", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
                if st.button("Run pingouin T-test"):
                    sub = chosen_df[[colA,colB]].dropna()
                    if len(sub)>1:
                        result = pg.ttest(sub[colA], sub[colB], paired=True)
                        st.write(result)
                    else:
                        st.warning("Not enough data after NA drop.")
            else:
                st.info("Need >=2 numeric columns for T-test.")

    ########################################################################
    # TAB 4: Auto-Analysis (Product vs %)
    ########################################################################
    with tabs[3]:
        st.header("Auto-Analysis & Product vs. % Detection")
        df_names = [d["name"] for d in st.session_state["uploaded_dfs"]]
        sel = st.selectbox("Select DF for product->%", ["(None)"] + df_names)
        if sel!="(None)":
            df_ = None
            for dd in st.session_state["uploaded_dfs"]:
                if dd["name"]==sel:
                    df_ = dd["df"]
                    break
            if df_ is not None and not df_.empty:
                if st.button("Parse product->%"):
                    res = parse_product_percent_table(df_)
                    if res.empty:
                        st.warning("No product->% found.")
                    else:
                        st.success(f"Found {len(res)} rows.")
                        st.dataframe(res)
                        fig_bar = px.bar(res, x="Product", y="Usage", text="Usage")
                        fig_bar.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                        fig_bar.update_yaxes(tickformat=".0%")
                        st.plotly_chart(fig_bar)

    ########################################################################
    # TAB 5: LLM Summaries
    ########################################################################
    with tabs[4]:
        st.header("LLM Summaries & Sentiment (T5, OpenAI, LLaMA)")

        all_text = "\n\n".join(st.session_state["parsed_text"])
        st.write(f"**Combined text length**={len(all_text)}")

        bullet_style = st.checkbox("Bullet style summary?", False)
        max_len = st.slider("max_length per chunk",50,300,120)
        min_len = st.slider("min_length per chunk",10,max_len-10,40)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Local T5 Summarize"):
                s = local_t5_summarize(all_text, max_length=max_len, min_length=min_len, bullet_points=bullet_style)
                st.write(s)

        with col2:
            openai_key = st.text_input("OpenAI API Key (for GPT)", value="", type="password")
            if st.button("OpenAI Summarize"):
                s = openai_summarize(all_text, openai_key)
                st.write(s)

        with col3:
            if LLAMA_AVAILABLE:
                llama_path = st.text_input("Local LLaMA model path", value="")
                if st.button("Local LLaMA Summarize"):
                    s = local_llama_summarize(all_text, llama_path)
                    st.write(s)
            else:
                st.info("llama-cpp not installed or build failed. See instructions to fix.")

        st.write("---")
        if st.button("Check Sentiment"):
            pol, subj = text_sentiment(all_text)
            st.write(f"**Polarity**={pol:.2f}, **Subjectivity**={subj:.2f}")

    ########################################################################
    # TAB 6: Table Extraction (Camelot/Tabula)
    ########################################################################
    with tabs[5]:
        st.header("Advanced PDF Table Extraction: Camelot or Tabula")
        pdf_file = st.file_uploader("Upload PDF for advanced table extraction", type=["pdf"])
        if pdf_file:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmpf:
                tmpf.write(pdf_file.read())
                tmpf.flush()
                pdf_path = tmpf.name
            if CAMELOT_AVAILABLE:
                if st.button("Extract with Camelot"):
                    c_tables = extract_tables_camelot(pdf_path)
                    st.write(f"Found {len(c_tables)} tables with Camelot.")
                    for i, cdf_ in enumerate(c_tables):
                        st.subheader(f"Table #{i+1}")
                        st.dataframe(cdf_)
                        # optionally store as new DF
            else:
                st.warning("Camelot not installed or import failed.")
            if TABULA_AVAILABLE:
                if st.button("Extract with Tabula"):
                    t_tables = extract_tables_tabula(pdf_path)
                    st.write(f"Found {len(t_tables)} tables with Tabula.")
                    for i, cdf_ in enumerate(t_tables):
                        st.subheader(f"Table #{i+1}")
                        st.dataframe(cdf_)
            else:
                st.warning("Tabula not installed or import failed.")
            try:
                os.remove(pdf_path)
            except:
                pass

    ########################################################################
    # TAB 7: R & SAS bridging
    ########################################################################
    with tabs[6]:
        st.header("R & SAS bridging")
        st.write("""
        You can run R or SAS code from here if rpy2 or saspy are installed and configured. 
        This is purely demonstration.
        """)
        r_code = st.text_area("Enter some R code", "R.version")
        if st.button("Run R Code"):
            result = run_r_code(r_code)
            st.write(result)

        st.write("---")
        sas_code = st.text_area("Enter some SAS code", "PROC MEANS DATA=SASHELP.CLASS;RUN;")
        if st.button("Run SAS Code"):
            result = run_sas_code(sas_code)
            st.write(result)

    ########################################################################
    # TAB 8: Embedding Search with FAISS
    ########################################################################
    with tabs[7]:
        st.header("Embedding-Based 'Google-Like' Search (FAISS + sentence-transformers)")

        # Combine text
        all_texts = st.session_state["parsed_text"]
        # Let user build or rebuild index
        if st.button("Build FAISS index from all text"):
            if not FAISS_AVAILABLE:
                st.warning("FAISS not installed. pip install faiss-cpu faiss-gpu or similar.")
            else:
                st.session_state["text_list"] = all_texts
                st.session_state["faiss_index"], st.session_state["embeddings"] = build_faiss_index(all_texts)
                st.success("Index built. You can now search.")

        query = st.text_input("Enter query for semantic search")
        top_k = st.slider("Top K results",1,10,3)
        if st.button("Search"):
            if not st.session_state["faiss_index"]:
                st.warning("No FAISS index built yet.")
            else:
                results = faiss_search(query, st.session_state["faiss_index"],
                                       st.session_state["embeddings"], st.session_state["text_list"], top_k)
                st.write("**Search Results**:")
                for r in results:
                    st.write(f"distance={r['distance']:.2f} => snippet: {r['text'][:300]}...")

    ########################################################################
    # TAB 9: External Data
    ########################################################################
    with tabs[8]:
        st.header("Fetch External Data (Reddit, NewsAPI, etc.)")
        st.write("Enter your credentials below:")
        reddit_id = st.text_input("Reddit Client ID", value=st.session_state["api_creds"].get("reddit_id",""))
        reddit_secret = st.text_input("Reddit Client Secret", value=st.session_state["api_creds"].get("reddit_secret",""), type="password")
        reddit_ua = st.text_input("Reddit UserAgent", value=st.session_state["api_creds"].get("reddit_ua",""))
        newsapi_key = st.text_input("NewsAPI Key", value=st.session_state["api_creds"].get("newsapi_key",""), type="password")

        if st.button("Save Credentials"):
            st.session_state["api_creds"]["reddit_id"] = reddit_id
            st.session_state["api_creds"]["reddit_secret"] = reddit_secret
            st.session_state["api_creds"]["reddit_ua"] = reddit_ua
            st.session_state["api_creds"]["newsapi_key"] = newsapi_key
            st.success("Saved in session.")

        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fetch from Reddit")
            sb = st.text_input("Subreddit", "news")
            lim = st.number_input("Limit",1,100,10)
            if st.button("Go Reddit"):
                df_r = fetch_reddit_data(sb, lim, reddit_id, reddit_secret, reddit_ua)
                if not df_r.empty:
                    st.dataframe(df_r)
                    st.session_state["uploaded_dfs"].append({"name":f"Reddit_{sb}", "df":df_r})
        with col2:
            st.subheader("Fetch from NewsAPI")
            q = st.text_input("Query for news", "technology")
            psize = st.number_input("page_size",1,100,10)
            if st.button("Go NewsAPI"):
                df_n = fetch_newsapi_data(q, newsapi_key, psize)
                if not df_n.empty:
                    st.dataframe(df_n)
                    st.session_state["uploaded_dfs"].append({"name":f"NewsAPI_{q}", "df":df_n})

    ########################################################################
    # TAB 10: DEPLOY INFO
    ########################################################################
    with tabs[9]:
        st.header("Deployment & Installation Instructions")
        st.markdown(r"""
**1. Install** everything:

```bash
pip install --upgrade pip setuptools wheel

pip install streamlit pandas numpy pdfplumber docx2txt PyPDF2 pillow opencv-python pytesseract \
            plotdigitizer torch transformers textblob scikit-learn plotly openpyxl requests \
            scipy statsmodels pingouin pyarrow""")


