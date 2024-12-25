################################################################################################
# ENHANCED FOOTCARE DASHBOARD
# -----------------------------------------------------------------------------------------------
# Key Features:
#   1) Multi-file upload & parse (PDF, Word, CSV, Excel, images).
#   2) ETL function that tries to parse "product -> percentage" from Excel for quick auto visuals.
#   3) All previous advanced functionalities (pivot, correlation, regression, chart extraction).
#   4) Local LLM summarization with T5.
#   5) Single file, no placeholders, ready for serious usage or investor demos.
################################################################################################

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

###############################################
# TESSERACT PATH (IF NEEDED ON WINDOWS)
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


###############################################
# 3. ETL: AUTO-PARSE “product vs. percentage” ROWS
###############################################
def parse_product_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to automatically detect rows with a product name + percentage usage.
    For example:
        Foot moisturizer   29%
        Shoe inserts       28%
    We do a best-effort approach:
      - If a cell looks like a string, next cell looks like .XX or XX%
      - Convert them into a two-column table: [Product, Value].
    Or if the columns are reversed, we try that too.

    Return a cleaned DataFrame with columns = ['Product','Usage'] if found.
    """
    # We gather possible product->value rows
    # Because user’s data might be in any columns.
    # We'll flatten the table row by row, checking pairs of cells.

    # 1) Flatten into (row_index, col_index, cell_value)
    all_cells = []
    for r in range(len(df)):
        for c in range(len(df.columns)):
            val = df.iloc[r, c]
            # Convert to string if not null
            if pd.isna(val):
                val = ""
            else:
                val = str(val).strip()
            all_cells.append((r, c, val))

    # 2) We attempt to pair consecutive columns in a row as "product, usage"
    # Because your sample has e.g. ( Foot moisturizer, 29% ) in row
    # We'll store potential matches in a list
    pairs = []
    for r in range(len(df)):
        row_vals = df.iloc[r].tolist()
        # row_vals is like [Foot moisturizer, 0.29, None, ...] or [Shoe inserts, 0.28, ...]
        # We skip empty
        cleaned = [str(x).strip() for x in row_vals if str(x).strip()]
        if len(cleaned) == 2:
            # possibly product->value
            p, v = cleaned
            # If v is something like '29%' or '0.29' or '29.0' etc
            # We'll parse it
            usage = parse_percentage(v)
            if usage is not None:
                pairs.append((p, usage))

    # If we found some pairs, let's return them as DF
    if pairs:
        out_df = pd.DataFrame(pairs, columns=["Product", "Usage"])
        return out_df

    # If that didn't work, let's try a slower approach:
    # We'll look for any row that has >=2 non-empty cells
    # The first is text, second is numeric/percent
    possible_rows = []
    for r in range(len(df)):
        row_vals = [str(x).strip() for x in df.iloc[r].tolist() if str(x).strip()]
        # If it has at least 2 items
        if len(row_vals) >= 2:
            # Attempt the last item as usage
            usage_val = parse_percentage(row_vals[-1])
            # The rest joined as product name?
            if usage_val is not None:
                product_str = " ".join(row_vals[:-1])
                possible_rows.append((product_str, usage_val))
    if possible_rows:
        out_df = pd.DataFrame(possible_rows, columns=["Product", "Usage"])
        return out_df

    # Return empty if no luck
    return pd.DataFrame()


def parse_percentage(val: str):
    """
    Convert a string like '29%', '0.29', '0.28', '28.0' to numeric float.
    E.g. '29%' -> 0.29 (i.e. 29%)
    E.g. '0.29' -> 0.29
    E.g. '29'   -> 0.29 (assuming it means 29%)
    If it doesn't parse, return None.
    """
    val = val.strip()
    # If ends with '%'
    if val.endswith('%'):
        try:
            num = float(val[:-1])
            return num / 100.0
        except:
            return None
    else:
        # try float
        try:
            f = float(val)
            # if f>1.0, assume f=29 means 29%
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
    if not summarizer:
        return "ERROR: T5 summarizer not loaded or installed."
    if not text.strip():
        return "No text to summarize."
    chunk_size = 1000
    pieces = []
    for i in range(0, len(text), chunk_size):
        c = text[i:i + chunk_size]
        out = summarizer(c, max_length=max_length, min_length=min_length, do_sample=False)
        pieces.append(out[0]["summary_text"])
    return "\n".join(pieces)


def sentiment_of_text(text: str):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


###############################################
# 5. STREAMLIT APP
###############################################
def main():
    st.set_page_config(page_title="Enhanced Footcare Dashboard", layout="wide")

    st.title("Enhanced Footcare Dashboard")
    st.write("""
    **Upgraded** to handle:
    1. Automatic detection of "product vs. percentage" rows in Excel (or CSV).
    2. Creates a quick bar chart or other visuals from that data.
    3. Continues to offer advanced pivot tables, correlation, local LLM summaries, etc.

    No placeholders, all in one file. Let's go!
    """)

    tabs = st.tabs(["Upload & ETL", "Auto-Analysis (Product vs. %)", "Advanced Visuals", "Analytics", "LLM Summaries",
                    "Deploy Info"])

    # SESSION KEYS
    if "parsed_text" not in st.session_state:
        st.session_state["parsed_text"] = []
    if "chart_dfs" not in st.session_state:
        st.session_state["chart_dfs"] = []
    if "uploaded_dfs" not in st.session_state:
        st.session_state["uploaded_dfs"] = []

    ###############################################
    # TAB 1: Upload & ETL
    ###############################################
    with tabs[0]:
        st.header("Upload & Parse Files (XLSX, CSV, PDF, DOCX, etc.)")

        files = st.file_uploader("Drop your files here", accept_multiple_files=True)

        if files:
            if st.button("Process Files"):
                st.session_state["parsed_text"] = []
                st.session_state["chart_dfs"] = []
                st.session_state["uploaded_dfs"] = []

                for f in files:
                    fname = f.name
                    st.subheader(f"Processing: {fname}")

                    # PDF
                    if fname.lower().endswith(".pdf"):
                        # extract text
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
                        # images -> detect chart
                        st.write("Extracting images from PDF...")
                        images = extract_images_from_pdf(path_pdf)
                        if images:
                            st.info(f"Found {len(images)} images.")
                            for i, img in enumerate(images):
                                maybe_chart = detect_chart_in_image(img)
                                if maybe_chart:
                                    st.write(
                                        f"Chart-like image found (Image #{i + 1}). Attempting data digitization...")
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
                        maybe_chart = detect_chart_in_image(pil_img)
                        if maybe_chart:
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
                st.write("Now check out other tabs for analysis, auto-charts, etc.")

    ###############################################
    # TAB 2: Auto-Analysis
    ###############################################
    with tabs[1]:
        st.header("Auto-Analysis: Product vs. Percentage")
        st.write(
            "If your XLSX/CSV has rows like 'Foot moisturizer' '29%' in two adjacent cells, we can auto-build a bar chart.")

        # Let user select from the loaded DataFrames
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

                        # Let's also do a quick bar chart
                        fig = px.bar(result_df, x="Product", y="Usage", title="Auto-Extracted Usage", text="Usage")
                        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                        fig.update_yaxes(tickformat=".0%")  # show y axis as percent
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                        # Possibly we want to sort by descending usage
                        st.write("**Sorted by usage (descending):**")
                        sorted_df = result_df.sort_values("Usage", ascending=False).reset_index(drop=True)
                        st.dataframe(sorted_df)
            else:
                st.info("That DataFrame is empty or not found. Try uploading a different file.")

    ###############################################
    # TAB 3: Advanced Visuals
    ###############################################
    with tabs[2]:
        st.header("Advanced Visuals (2D, 3D, Hist, etc.)")

        # Combine all possible data sources: uploaded + chart extraction
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
            st.info("No data available. Upload & parse first in 'Upload & ETL' tab.")

    ###############################################
    # TAB 4: Analytics
    ###############################################
    with tabs[3]:
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
                                                 ["None", "Pivot Table", "Correlation Heatmap", "Regression",
                                                  "K-Means Clustering"])
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
                                    st.write(f"Slope={slope}, Intercept={intercept}")
                                    # Plot
                                    fig_reg = px.scatter(subdf, x=xcol, y=ycol)
                                    xline = np.linspace(subdf[xcol].min(), subdf[xcol].max(), 50)
                                    yline = slope * xline + intercept
                                    fig_reg.add_trace(go.Scatter(x=xline, y=yline, mode='lines', name='Regression'))
                                    st.plotly_chart(fig_reg)

                    elif analysis_type == "K-Means Clustering":
                        numeric_cols = [c for c in chosen_df.columns if pd.api.types.is_numeric_dtype(chosen_df[c])]
                        cluster_cols = st.multiselect("Columns for clustering", numeric_cols)
                        k = st.slider("Number of Clusters", 2, 10, 3)
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
                                    fig_k3 = px.scatter_3d(subdf, x=cluster_cols[0], y=cluster_cols[1],
                                                           z=cluster_cols[2], color="Cluster")
                                    st.plotly_chart(fig_k3)
                else:
                    st.warning("Chosen DataFrame is empty or not found.")
        else:
            st.info("No data available. Upload & parse first.")

    ###############################################
    # TAB 5: LLM Summaries
    ###############################################
    with tabs[4]:
        st.header("LLM Summaries (Local T5)")
        all_text = "\n\n".join(st.session_state["parsed_text"])
        st.write(
            f"We have {len(st.session_state['parsed_text'])} text chunks available. Combined length={len(all_text)} chars.")
        if all_text:
            if st.button("Summarize All Extracted Text"):
                with st.spinner("Summarizing..."):
                    summary = local_summarize_text(all_text)
                st.success("Done!")
                st.write("### Summary:")
                st.write(summary)
                # Sentiment
                pol, subj = sentiment_of_text(all_text)
                st.write(f"**Overall Sentiment**: Polarity={pol:.2f}, Subjectivity={subj:.2f}")

    ###############################################
    # TAB 6: Deploy Info
    ###############################################
    with tabs[5]:
        st.header("Deployment Info")
        st.markdown("""
        **How to Deploy as a Website**:
        1. **Streamlit Cloud**: push code to GitHub, connect on [streamlit.io/cloud](https://streamlit.io/cloud).
        2. **Docker**: Create a Dockerfile, `pip install -r requirements.txt`, `streamlit run enhanced_footcare_dashboard.py`.
        3. **Local**: Just run `streamlit run enhanced_footcare_dashboard.py`.

        **Dependencies** (example):
        ```bash
        pip install streamlit pandas numpy pdfplumber docx2txt PyPDF2 pillow opencv-python pytesseract \
                    plotdigitizer torch transformers textblob scikit-learn plotly openpyxl
        ```
        - Also ensure Tesseract is installed system-wide if you want OCR-based chart detection.
        """)


if __name__ == "__main__":
    main()
