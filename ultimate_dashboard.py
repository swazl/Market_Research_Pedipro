################################################################################
# ULTIMATE MARKET RESEARCH DASHBOARD
#
# Key Goals:
# 1. ETL: Ingest, transform, and clean data (CSV, Excel, PDF, Word, images, plus manual text).
# 2. NLP: Summarize text (T5), sentiment analysis (TextBlob).
# 3. Visualization: 2D/3D plots, advanced analytics (pivot, correlation, regression, clustering).
# 4. PDF Chart Extraction: Attempt to auto-digitize chart images in PDFs.
# 5. Free/Open-Source Only: No paid API keys required.
# 6. Simple Enough for C-Suite: Natural language usage where possible.
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import docx2txt
import PyPDF2
import os
import io
import re
import tempfile
import cv2
import pytesseract
from PIL import Image
from typing import List

# Chart digitizer
import plotdigitizer

# Local LLM Summaries (T5)
import torch
from transformers import pipeline

# Text analysis
from textblob import TextBlob

# ML
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Plotly for interactive visuals
import plotly.express as px
import plotly.graph_objects as go

# Optional: for free data sources like Google Trends or Stocks (uncomment if desired)
# ! pip install pytrends yfinance
# from pytrends.request import TrendReq
# import yfinance as yf


###############################################
# If needed on Windows, set tesseract path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
###############################################

###############################################
# 0. LOCAL LLM SUMMARIZER
###############################################
try:
    summarizer = pipeline("summarization", model="t5-small", device=torch.device("cpu"))
except Exception:
    summarizer = None


###############################################
# 1. FILE READ HELPERS
###############################################
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


def read_csv_to_df(file_obj) -> pd.DataFrame:
    return pd.read_csv(file_obj)


def read_excel_to_df(file_obj) -> pd.DataFrame:
    import openpyxl  # ensure it's installed
    return pd.read_excel(file_obj)


def is_image_file(filename: str) -> bool:
    ext = filename.lower().split(".")[-1]
    return ext in ["png", "jpg", "jpeg", "bmp", "tiff", "gif"]


###############################################
# 2. CHART DETECTION & DIGITIZATION (PDF IMAGES)
###############################################
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

    # Heuristic: if we see chart-like words OR lots of numeric content, plus edges
    if (found_kw or numeric_count > 5) and edgecount > 500:
        return True
    return False


def digitize_chart(pil_img: Image.Image, plot_type="scatter") -> pd.DataFrame:
    """
    Use plotdigitizer to attempt extracting X,Y from an image.
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


###############################################
# 3. ETL: AUTO-PARSE “product vs. percentage”
###############################################
def parse_product_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to automatically detect rows with a product name + percentage usage.
    For example:
        Foot moisturizer   29%
        Shoe inserts       28%
    We do a best-effort approach:
      - If a row has 2 cells, where the second cell is a parseable percentage, build a table.
      - If that fails, we look for the last cell in a row to be a percentage and the rest combined into product.
    Return a cleaned DataFrame with columns = ['Product','Usage'] if found.
    """
    # 1) Try easy approach: row with exactly 2 items => (product, usage)
    pairs = []
    for _, row in df.iterrows():
        cleaned = [str(x).strip() for x in row if str(x).strip()]
        if len(cleaned) == 2:
            p, v = cleaned
            usage = parse_percentage(v)
            if usage is not None:
                pairs.append((p, usage))

    if pairs:
        out_df = pd.DataFrame(pairs, columns=["Product", "Usage"])
        return out_df

    # 2) If that fails, try last-cell approach
    possible_rows = []
    for _, row in df.iterrows():
        cleaned = [str(x).strip() for x in row if str(x).strip()]
        if len(cleaned) >= 2:
            usage_val = parse_percentage(cleaned[-1])
            if usage_val is not None:
                product_str = " ".join(cleaned[:-1])
                possible_rows.append((product_str, usage_val))
    if possible_rows:
        out_df = pd.DataFrame(possible_rows, columns=["Product", "Usage"])
        return out_df

    return pd.DataFrame()  # empty if not found


def parse_percentage(val: str):
    """
    Convert '29%', '0.29', '29.0' to float(0.29). Return None if not parseable.
    """
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


###############################################
# 4. LLM + SENTIMENT
###############################################
def local_summarize_text(text, max_length=120, min_length=40):
    """
    Summarize text in ~1000-char chunks with T5. Return combined summary.
    """
    if not summarizer:
        return "ERROR: T5 summarizer not loaded or installed. Check your environment."
    text = text.strip()
    if not text:
        return "No text to summarize."
    chunk_size = 1000
    pieces = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size]
        out = summarizer(chunk_text, max_length=max_length, min_length=min_length, do_sample=False)
        pieces.append(out[0]["summary_text"])
    return "\n".join(pieces)


def sentiment_of_text(text: str):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


###############################################
# 5. STREAMLIT APP
###############################################
def main():
    st.set_page_config(page_title="Ultimate Market Research Dashboard", layout="wide")

    st.title("Ultimate Market Research Dashboard")
    st.write("""
    **Key Features**:
    1. Upload & parse data/text from multiple sources (PDF, CSV, Excel, DOCX, images).
    2. Basic ETL transformations (rename columns, drop duplicates, fill missing).
    3. Automatic detection of "product vs. percentage" rows for quick bar charts.
    4. Advanced analytics (pivot, correlation, regression, clustering).
    5. Local T5 Summaries & sentiment analysis.
    6. PDF chart detection & digitization.
    7. (Optional) Pull data from free sources like Google Trends or Yahoo Finance (uncomment code).
    """)

    # Initialize session keys
    if "parsed_text" not in st.session_state:
        st.session_state["parsed_text"] = []
    if "chart_dfs" not in st.session_state:
        st.session_state["chart_dfs"] = []
    if "uploaded_dfs" not in st.session_state:
        st.session_state["uploaded_dfs"] = []

    # Multi-Tab Layout
    tabs = st.tabs([
        "Upload & Parse",
        "ETL & Manual Input",
        "Auto-Analysis",
        "Advanced Visuals",
        "Analytics",
        "LLM Summaries",
        "Deploy Info"
    ])

    ################################################
    # TAB 1: UPLOAD & PARSE
    ################################################
    with tabs[0]:
        st.header("Upload & Parse Files (XLSX, CSV, PDF, DOCX, images)")
        files = st.file_uploader("Drop your files here", accept_multiple_files=True)

        if files and st.button("Process Files"):
            # Reset session storage if re-processing
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
                    cdf = read_csv_to_df(f)
                    st.session_state["uploaded_dfs"].append({"name": fname, "df": cdf})
                    st.write("Sample of CSV:")
                    st.dataframe(cdf.head())

                # XLSX
                elif fname.lower().endswith(".xlsx"):
                    xdf = read_excel_to_df(f)
                    st.session_state["uploaded_dfs"].append({"name": fname, "df": xdf})
                    st.write("Sample of Excel:")
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

            st.success("All files processed!")
            st.write("Check other tabs for ETL, analysis, and more.")

    ################################################
    # TAB 2: ETL & MANUAL INPUT
    ################################################
    with tabs[1]:
        st.header("ETL & Manual Data Input")
        st.write("Perform basic cleaning steps and/or add manual text/data if not using an API.")

        # 2.1 Show existing DFs, allow some transformations
        df_list = [d["name"] for d in st.session_state["uploaded_dfs"]]
        if df_list:
            st.subheader("Basic DataFrame Transformations")

            chosen_df_name = st.selectbox("Select DataFrame to transform", options=["(None)"] + df_list)
            if chosen_df_name != "(None)":
                # find the DF
                chosen_df = None
                for d in st.session_state["uploaded_dfs"]:
                    if d["name"] == chosen_df_name:
                        chosen_df = d["df"]
                        break

                if chosen_df is not None:
                    st.write("Preview:")
                    st.dataframe(chosen_df.head())

                    # rename columns
                    columns = list(chosen_df.columns)
                    rename_col = st.selectbox("Column to rename", options=["(None)"] + columns)
                    new_name = st.text_input("New column name")
                    if st.button("Rename Column"):
                        if rename_col != "(None)" and new_name.strip():
                            chosen_df.rename(columns={rename_col: new_name.strip()}, inplace=True)
                            st.success(f"Renamed {rename_col} -> {new_name.strip()}")

                    # drop duplicates
                    if st.button("Drop Duplicates"):
                        before = len(chosen_df)
                        chosen_df.drop_duplicates(inplace=True)
                        after = len(chosen_df)
                        st.success(f"Dropped {before - after} duplicates.")

                    # fill missing
                    fill_method = st.selectbox("Fill missing with", ["(None)", "Forward Fill", "0", "Mean of column"])
                    if fill_method != "(None)":
                        if st.button("Apply Fill"):
                            if fill_method == "Forward Fill":
                                chosen_df.fillna(method="ffill", inplace=True)
                            elif fill_method == "0":
                                chosen_df.fillna(0, inplace=True)
                            elif fill_method == "Mean of column":
                                for c in chosen_df.select_dtypes(include=[np.number]).columns:
                                    chosen_df[c].fillna(chosen_df[c].mean(), inplace=True)
                            st.success(f"Applied fill method: {fill_method}")

                    st.write("Updated Preview:")
                    st.dataframe(chosen_df.head(len(chosen_df)))

        # 2.2 Manual text input
        st.subheader("Manual Text Entry")
        manual_text = st.text_area("Paste or type text here (e.g., from IBISWorld or MarketResearch.com reports):")
        if st.button("Add to Text Corpus"):
            if manual_text.strip():
                st.session_state["parsed_text"].append(manual_text.strip())
                st.success(f"Added {len(manual_text.strip())} characters to text corpus.")
            else:
                st.warning("No text entered.")

    ################################################
    # TAB 3: Auto-Analysis (Product vs. %)
    ################################################
    with tabs[2]:
        st.header("Auto-Analysis: Product vs. Percentage")
        st.write("For data like 'Product -> 25%', quickly generate bar charts.")

        df_choices = [d["name"] for d in st.session_state["uploaded_dfs"]]
        selected_df = st.selectbox("Select an uploaded DataFrame", options=["(None)"] + df_choices)
        if selected_df != "(None)":
            # find the actual DF
            chosen_df = None
            for d in st.session_state["uploaded_dfs"]:
                if d["name"] == selected_df:
                    chosen_df = d["df"]
                    break

            if chosen_df is not None and not chosen_df.empty:
                st.write("Preview of the raw DataFrame:")
                st.dataframe(chosen_df.head(len(chosen_df)))

                if st.button("Auto-Parse Product/Percent"):
                    result_df = parse_product_percent_table(chosen_df)
                    if result_df.empty:
                        st.warning("Could not find any 'product -> percentage' rows.")
                    else:
                        st.success(f"Parsed {len(result_df)} rows of product usage!")
                        st.dataframe(result_df)

                        # Quick bar chart
                        fig = px.bar(result_df, x="Product", y="Usage", title="Auto-Extracted Usage", text="Usage")
                        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                        fig.update_yaxes(tickformat=".0%")  # show y axis as percent
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                        # Sorted table
                        st.write("**Sorted by Usage (Descending):**")
                        sorted_df = result_df.sort_values("Usage", ascending=False).reset_index(drop=True)
                        st.dataframe(sorted_df)
            else:
                st.info("That DataFrame is empty or not found. Try uploading a different file.")

    ################################################
    # TAB 4: Advanced Visuals
    ################################################
    with tabs[3]:
        st.header("Advanced Visuals (2D/3D, Hist, etc.)")

        all_data_options = []
        for dd in st.session_state["uploaded_dfs"]:
            all_data_options.append(dd["name"])
        for i, cdf in enumerate(st.session_state["chart_dfs"]):
            all_data_options.append(f"ChartDF #{i + 1}")

        if all_data_options:
            chosen_name = st.selectbox("Select a DataFrame", ["(None)"] + all_data_options)
            if chosen_name != "(None)":
                # get DF
                chosen_df = None
                if chosen_name.startswith("ChartDF"):
                    idx = int(chosen_name.split("#")[1]) - 1
                    chosen_df = st.session_state["chart_dfs"][idx]
                else:
                    for dd in st.session_state["uploaded_dfs"]:
                        if dd["name"] == chosen_name:
                            chosen_df = dd["df"]
                            break

                if chosen_df is not None and not chosen_df.empty:
                    st.write("Data Preview:")
                    st.dataframe(chosen_df.head())

                    # Visualization config
                    numeric_cols = [c for c in chosen_df.columns if pd.api.types.is_numeric_dtype(chosen_df[c])]
                    if numeric_cols:
                        vtype = st.selectbox("Plot Type",
                                             ["Scatter 2D", "Scatter 3D", "Histogram", "Line Chart", "Box Plot"])
                        if vtype == "Scatter 2D":
                            xcol = st.selectbox("X-axis", numeric_cols)
                            ycol = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
                            color_choice = st.selectbox("Color By (optional)", [None] + list(chosen_df.columns))
                            if st.button("Create 2D Scatter"):
                                fig = px.scatter(chosen_df, x=xcol, y=ycol,
                                                 color=None if color_choice in (None, "") else color_choice)
                                st.plotly_chart(fig, use_container_width=True)

                        elif vtype == "Scatter 3D":
                            if len(numeric_cols) < 3:
                                st.warning("Need >=3 numeric columns for 3D scatter.")
                            else:
                                xcol = st.selectbox("X-axis", numeric_cols)
                                ycol = st.selectbox("Y-axis", numeric_cols)
                                zcol = st.selectbox("Z-axis", numeric_cols)
                                color_choice = st.selectbox("Color By (optional)", [None] + list(chosen_df.columns))
                                if st.button("Create 3D Scatter"):
                                    fig = px.scatter_3d(chosen_df, x=xcol, y=ycol, z=zcol,
                                                        color=None if color_choice in (None, "") else color_choice)
                                    st.plotly_chart(fig, use_container_width=True)

                        elif vtype == "Histogram":
                            hist_col = st.selectbox("Column", numeric_cols)
                            bins = st.slider("Bins", 5, 50, 20)
                            if st.button("Create Histogram"):
                                fig = px.histogram(chosen_df, x=hist_col, nbins=bins)
                                st.plotly_chart(fig, use_container_width=True)

                        elif vtype == "Line Chart":
                            line_x = st.selectbox("X-axis", numeric_cols)
                            line_y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
                            if st.button("Create Line Chart"):
                                fig = px.line(chosen_df, x=line_x, y=line_y)
                                st.plotly_chart(fig, use_container_width=True)

                        elif vtype == "Box Plot":
                            box_col = st.selectbox("Column", numeric_cols)
                            if st.button("Create Box Plot"):
                                fig = px.box(chosen_df, y=box_col)
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numeric columns found in this DataFrame.")
                else:
                    st.warning("Chosen DataFrame is empty or not found.")
        else:
            st.info("No data available. Upload & parse first.")

    ################################################
    # TAB 5: Analytics
    ################################################
    with tabs[4]:
        st.header("Analytics (Pivot, Correlation, Regression, Clustering)")
        all_data_options = []
        for dd in st.session_state["uploaded_dfs"]:
            all_data_options.append(dd["name"])
        for i, cdf in enumerate(st.session_state["chart_dfs"]):
            all_data_options.append(f"ChartDF #{i + 1}")

        if all_data_options:
            an_choice = st.selectbox("Select DataFrame", ["(None)"] + all_data_options)
            if an_choice != "(None)":
                chosen_df = None
                if an_choice.startswith("ChartDF"):
                    idx = int(an_choice.split("#")[1]) - 1
                    chosen_df = st.session_state["chart_dfs"][idx]
                else:
                    for dd in st.session_state["uploaded_dfs"]:
                        if dd["name"] == an_choice:
                            chosen_df = dd["df"]
                            break

                if chosen_df is not None and not chosen_df.empty:
                    st.write("Data Preview:")
                    st.dataframe(chosen_df.head())

                    analysis_type = st.selectbox("Choose Analysis",
                                                 ["None", "Pivot Table", "Correlation Heatmap",
                                                  "Regression", "K-Means Clustering"])

                    if analysis_type == "Pivot Table":
                        cols = list(chosen_df.columns)
                        if len(cols) < 2:
                            st.warning("Need at least 2 columns for pivot.")
                        else:
                            idx_col = st.selectbox("Pivot Index", cols)
                            val_col = st.selectbox("Pivot Values (numeric)", cols)
                            agg_func = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"])
                            if st.button("Create Pivot Table"):
                                try:
                                    pvt = chosen_df.pivot_table(index=idx_col, values=val_col, aggfunc=agg_func)
                                    st.write("Pivot Table:")
                                    st.dataframe(pvt)
                                except Exception as e:
                                    st.error(f"Pivot error: {e}")

                    elif analysis_type == "Correlation Heatmap":
                        numeric_df = chosen_df.select_dtypes(include=[np.number])
                        if numeric_df.empty:
                            st.warning("No numeric columns.")
                        else:
                            corr = numeric_df.corr()
                            st.write("Correlation Matrix:")
                            st.dataframe(corr)
                            fig_heat = go.Figure(data=go.Heatmap(
                                z=corr.values,
                                x=corr.columns,
                                y=corr.columns,
                                colorscale='RdBu',
                                zmid=0
                            ))
                            st.plotly_chart(fig_heat, use_container_width=True)

                    elif analysis_type == "Regression":
                        numeric_cols = [c for c in chosen_df.columns if pd.api.types.is_numeric_dtype(chosen_df[c])]
                        if len(numeric_cols) < 2:
                            st.warning("Need >=2 numeric columns.")
                        else:
                            xcol = st.selectbox("X", numeric_cols)
                            ycol = st.selectbox("Y", numeric_cols, index=min(1, len(numeric_cols) - 1))
                            if st.button("Run Regression"):
                                subdf = chosen_df[[xcol, ycol]].dropna()
                                if subdf.empty:
                                    st.error("No data after dropping NAs.")
                                else:
                                    X = subdf[xcol].values.reshape(-1, 1)
                                    Y = subdf[ycol].values
                                    model = LinearRegression()
                                    model.fit(X, Y)
                                    slope = model.coef_[0]
                                    intercept = model.intercept_
                                    st.write(f"Slope = {slope}, Intercept = {intercept}")
                                    # Plot regression line
                                    fig_reg = px.scatter(subdf, x=xcol, y=ycol, title="Regression")
                                    xline = np.linspace(subdf[xcol].min(), subdf[xcol].max(), 50)
                                    yline = slope * xline + intercept
                                    fig_reg.add_trace(go.Scatter(x=xline, y=yline, mode='lines', name='Regression'))
                                    st.plotly_chart(fig_reg)

                    elif analysis_type == "K-Means Clustering":
                        numeric_cols = [c for c in chosen_df.columns if pd.api.types.is_numeric_dtype(chosen_df[c])]
                        cluster_cols = st.multiselect("Columns for clustering", numeric_cols)
                        k = st.slider("Number of Clusters (k)", 2, 10, 3)
                        if st.button("Run K-Means"):
                            subdf = chosen_df[cluster_cols].dropna()
                            if subdf.empty:
                                st.error("No valid data for clustering.")
                            else:
                                km = KMeans(n_clusters=k, random_state=42)
                                labels = km.fit_predict(subdf)
                                subdf["Cluster"] = labels
                                st.dataframe(subdf.head(20))
                                if len(cluster_cols) == 2:
                                    fig_k2 = px.scatter(subdf, x=cluster_cols[0], y=cluster_cols[1], color="Cluster")
                                    st.plotly_chart(fig_k2)
                                elif len(cluster_cols) == 3:
                                    fig_k3 = px.scatter_3d(
                                        subdf, x=cluster_cols[0], y=cluster_cols[1],
                                        z=cluster_cols[2], color="Cluster"
                                    )
                                    st.plotly_chart(fig_k3)
                else:
                    st.warning("Chosen DataFrame is empty or not found.")
        else:
            st.info("No data available. Upload & parse first.")

    ################################################
    # TAB 6: LLM Summaries
    ################################################
    with tabs[5]:
        st.header("LLM Summaries (Local T5)")

        # Combine all parsed text
        all_text = "\n\n".join(st.session_state["parsed_text"])
        st.write(
            f"We have {len(st.session_state['parsed_text'])} text chunks. "
            f"Combined length={len(all_text)} characters."
        )

        if all_text:
            if st.button("Summarize All Text"):
                with st.spinner("Summarizing..."):
                    summary = local_summarize_text(all_text)
                st.success("Done!")
                st.write("### Summary:")
                st.write(summary)

                # Sentiment
                pol, subj = sentiment_of_text(all_text)
                st.write(f"**Overall Sentiment**: Polarity={pol:.2f}, Subjectivity={subj:.2f}")
        else:
            st.info("No text found. Upload PDF/DOCX or manually add text in 'ETL & Manual Input' tab.")

    ################################################
    # TAB 7: Deploy Info
    ################################################
    with tabs[6]:
        st.header("Deployment Info")
        st.markdown(r"""
**How to Deploy**:
1. **Local (PyCharm, VSCode, etc.)**  
   - Make sure you install libraries:  
     ```bash
     pip install streamlit pandas numpy pdfplumber docx2txt PyPDF2 pillow opencv-python pytesseract plotdigitizer \
                 torch transformers textblob scikit-learn plotly openpyxl
     ```
   - Then run:
     ```bash
     streamlit run ultimate_dashboard.py
     ```
   - Open the URL Streamlit provides (usually `http://localhost:8501`).

2. **Streamlit Cloud**  
   - Push `ultimate_dashboard.py` + `requirements.txt` to GitHub.  
   - Connect repo to [streamlit.io/cloud](https://streamlit.io/cloud).  

3. **Docker**  
   - Create a Dockerfile that installs Python + dependencies.  
   - `docker build -t ultimate_dashboard .`  
   - `docker run -p 8501:8501 ultimate_dashboard`  
   - Access `http://localhost:8501`.

**Note**:
- If chart detection doesn’t work on Windows, check Tesseract is installed properly and update the path in the code.
- For advanced data collection, you can install `pytrends` or `yfinance` for free data on Google Trends or stocks.
- For bigger LLMs, consider GPT4All or Llama 2 local models, but those require more setup.

Enjoy your *Ultimate Market Research Dashboard!* 
        """)


if __name__ == "__main__":
    main()
