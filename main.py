"""
Spectacular Market Research Tool
--------------------------------
This code is a proof of concept for a commercial-grade program that:
1. Ingests data (manual upload and potential free-tier APIs).
2. ETL: Extract, Transform, Load.
3. NLP with open-source or free-tier LLMs.
4. Visualization (2D/3D interactive).
5. Executive summary generation for IBIS / MarketResearch style reports.
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import PyPDF2  # or pdfplumber
import requests


# If you install transformers, you can do the summarization locally
# from transformers import pipeline

# -------------- 1. HELPER FUNCTIONS --------------

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple transformation function:
    - Drop null rows
    - Convert columns to appropriate dtypes
    - Return cleaned data
    """
    # Drop rows where all elements are NaN
    df.dropna(how='all', inplace=True)
    # Example: fill numerical columns with 0 if NA
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(0, inplace=True)
    return df


def load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF using PyPDF2.
    For large PDFs, you'd want chunking or more advanced logic.
    """
    text_content = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_content.append(page.extract_text())
    return "\n".join(text_content)


def free_api_demo():
    """
    Example of how you might load data from a free (or free-tier) API.
    We'll just do a simple GET request to a public data endpoint.
    This function is just a placeholder.
    """
    url = "https://jsonplaceholder.typicode.com/posts"  # A test free endpoint
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None


def summarize_text(text: str) -> str:
    """
    Placeholder for text summarization.
    Option 1: Using a local model with transformers pipeline.
    Option 2: Using Hugging Face Inference API if free tier is available.
    """

    # Here, we'll do a dummy summarizer
    # In production, you'd call a real model.
    # For demonstration, let's do a simple 'split' approach:
    if not text:
        return "No text to summarize."

    # TRIVIAL "summarizer"
    # This just takes the first 200 characters, obviously not a real summary.
    # Replace with advanced summarizer or HuggingFace pipeline for real usage.
    return text[:200] + "... [Truncated summary]"


# -------------- 2. MAIN WORKFLOW --------------

def main():
    print("\n--- Welcome to the Spectacular Market Research Tool ---\n")

    # Step A: Ask user how they'd like to provide data
    print("Data Ingestion Options:")
    print("1. Manual CSV upload")
    print("2. Manual PDF upload")
    print("3. Demo Free API call")
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        csv_path = input("Enter the CSV file path: ").strip()
        if not os.path.exists(csv_path):
            print("File does not exist.")
            sys.exit()
        df = load_csv(csv_path)
        df = transform_data(df)

        # Simple 2D visualization
        print("\nCreating a 2D scatter plot of the first two numeric columns (if available)...")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="2D Scatter Plot")
            fig.show()
        else:
            print("Not enough numeric columns to plot a scatter graph.")

    elif choice == "2":
        pdf_path = input("Enter the PDF file path: ").strip()
        if not os.path.exists(pdf_path):
            print("File does not exist.")
            sys.exit()
        pdf_text = load_pdf(pdf_path)
        summary = summarize_text(pdf_text)
        print("\n--- Executive Summary of PDF ---\n")
        print(summary)

    elif choice == "3":
        data = free_api_demo()
        if data:
            # Convert to a DF for demonstration
            df = pd.DataFrame(data)
            # Show a 3D Visualization if we have enough numeric data
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numeric_cols) >= 3:
                fig = px.scatter_3d(df,
                                    x=numeric_cols[0],
                                    y=numeric_cols[1],
                                    z=numeric_cols[2],
                                    title="3D Scatter from Free API")
                fig.show()
            else:
                print("Not enough numeric columns in the fetched data to show a 3D graph.")
        else:
            print("Failed to fetch data from free API.")
    else:
        print("Invalid choice. Exiting.")

    print("\n--- End of Program ---\n")


if __name__ == "__main__":
    main()
