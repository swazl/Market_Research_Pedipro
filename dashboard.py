import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from datetime import datetime
from pytrends.request import TrendReq
import praw
import openai
import logging
from functools import lru_cache

#########################
# CREDENTIALS (INLINE)
#########################

# Reddit Credentials
REDDIT_CLIENT_ID = "JQI8S3mbi5EHvZCUtyq_tLQ"
REDDIT_CLIENT_SECRET = "KI8QRmoJUZtAOPmQTnxztHiY8bJeSg"
# A more descriptive user agent often fixes 401 issues on Reddit
REDDIT_USER_AGENT = "PediproApp/1.0 by u/BeneficialLeague5233"

# YouTube Credentials
YOUTUBE_API_KEY = "AIzaSyD7SwzStMs_Job95bLl8c5G6IAYWUmBd10"

# NewsAPI
NEWS_API_KEY = "7c032cdf5e0f4045966f1ed40ff47d8a"

# Google Programmable Search Engine
GOOGLE_SEARCH_API_KEY = "AIzaSyD7SwzStMs_Job95bLl8c5G6IAYWUmBd10"
GOOGLE_SEARCH_ENGINE_ID = "a0f3ff8dee296466b"

# Yelp
YELP_API_KEY = (
    "CUmxP8aJu_sU_ge5IBM6nBqOKo3u_JFPCntbWVx4dLj22Kl_AW57OQu3HYczNY20Ytwv2NB_Up0-"
    "GC2454rthRv4rX6yoIZfJUMGc1XkxWXZrxuH6Kt4LegVmZ3Yx"
)

# Stack Exchange
STACKEXCHANGE_CLIENT_ID = "30445"
STACKEXCHANGE_CLIENT_SECRET = "zBR48JQG93jGGg2pT5NOg(("
STACKEXCHANGE_KEY = "rl_vkBp2zjpBjHfw5W5r4euiveS"

# Mediastack
MEDIASTACK_API_KEY = "YOUR_MEDIASTACK_API_KEY"

# GPT Key (OpenAI)
OPENAI_API_KEY = (
    "sk-proj-7mrZSh-GcA2xxidja9xXIcbWOzoioDlJvfCC9Qx5MmCvIc5CjOmA57ARBigKZT4sy-3jiz"
    "DpG9T3BlbkFJepTWk0TamwAex_zx4HB7mzSGILhBQhbHGEVCuST_lCAI3hZ4p9cQvO2Hmz-K8lLxtuo9PlOz0A"
)

# If you have an ngrok authtoken, here's where you'd store it if needed:
NGROK_AUTHTOKEN = "2qWwn7jD5StaxOxaChERLmL2moZ_2RHThVTWebxctnJhHfsT4"

# Set the OpenAI key
openai.api_key = OPENAI_API_KEY

#########################
# LOGGING
#########################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#########################
# DATA FETCHING FUNCTIONS (WITH LRU CACHE)
#########################

@lru_cache(maxsize=128)
def fetch_reddit_data(keyword, limit=20):
    logging.info(f"Fetching Reddit data for '{keyword}'...")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    posts = []
    try:
        for submission in reddit.subreddit("all").search(keyword, limit=limit):
            posts.append({
                "Source": "Reddit",
                "Keyword": keyword,
                "Title": submission.title,
                "Content": submission.selftext,
                "URL": submission.url,
                "Date": datetime.utcfromtimestamp(submission.created_utc).isoformat()
            })
    except Exception as e:
        logging.error(f"Error fetching Reddit data: {e}")
        st.error(f"Error fetching Reddit data: {e}")
    return pd.DataFrame(posts)

@lru_cache(maxsize=128)
def fetch_youtube_data(keyword, max_results=20):
    logging.info(f"Fetching YouTube data for '{keyword}'...")
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": keyword,
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY
    }
    results = []
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        for item in items:
            snippet = item["snippet"]
            video_id = item["id"].get("videoId", "")
            results.append({
                "Source": "YouTube",
                "Keyword": keyword,
                "Title": snippet.get("title"),
                "Content": snippet.get("description"),
                "URL": f"https://www.youtube.com/watch?v={video_id}" if video_id else None,
                "Date": snippet.get("publishedAt")
            })
    except Exception as e:
        logging.error(f"Error fetching YouTube data: {e}")
        st.error(f"Error fetching YouTube data: {e}")
    return pd.DataFrame(results)

@lru_cache(maxsize=128)
def fetch_news_data(keyword, page_size=20):
    logging.info(f"Fetching NewsAPI data for '{keyword}'...")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }
    df = pd.DataFrame()
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        results = []
        for art in articles:
            results.append({
                "Source": "NewsAPI",
                "Keyword": keyword,
                "Title": art.get("title"),
                "Content": art.get("description"),
                "URL": art.get("url"),
                "Date": art.get("publishedAt")
            })
        df = pd.DataFrame(results)
    except Exception as e:
        logging.error(f"Error fetching NewsAPI data: {e}")
        st.error(f"Error fetching NewsAPI data: {e}")
    return df

@lru_cache(maxsize=128)
def fetch_google_search_data(keyword, num=10):
    logging.info(f"Fetching Google Search data for '{keyword}'...")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": keyword,
        "num": num
    }
    df = pd.DataFrame()
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        results = []
        for item in items:
            results.append({
                "Source": "GoogleSearch",
                "Keyword": keyword,
                "Title": item.get("title"),
                "Content": item.get("snippet"),
                "URL": item.get("link"),
                "Date": None
            })
        df = pd.DataFrame(results)
    except Exception as e:
        logging.error(f"Error fetching GoogleSearch data: {e}")
        st.error(f"Error fetching GoogleSearch data: {e}")
    return df

@lru_cache(maxsize=128)
def fetch_yelp_data(keyword, location="New York", limit=20):
    logging.info(f"Fetching Yelp data for '{keyword}' in {location}...")
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    params = {
        "term": keyword,
        "location": location,
        "limit": limit
    }
    df = pd.DataFrame()
    try:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        businesses = data.get("businesses", [])
        results = []
        for biz in businesses:
            categories = [cat["title"] for cat in biz.get("categories", [])]
            results.append({
                "Source": "Yelp",
                "Keyword": keyword,
                "Title": biz.get("name"),
                "Content": ", ".join(categories),
                "URL": biz.get("url"),
                "Date": None
            })
        df = pd.DataFrame(results)
    except Exception as e:
        logging.error(f"Error fetching Yelp data: {e}")
        st.error(f"Error fetching Yelp data: {e}")
    return df

@lru_cache(maxsize=128)
def fetch_stackexchange_data(keyword, site="beauty", pagesize=20):
    logging.info(f"Fetching StackExchange data for '{keyword}' in site={site}...")
    url = "https://api.stackexchange.com/2.3/search"
    params = {
        "order": "desc",
        "sort": "activity",
        "intitle": keyword,
        "site": site,
        "pagesize": pagesize,
        "key": STACKEXCHANGE_KEY
    }
    df = pd.DataFrame()
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        results = []
        for item in items:
            results.append({
                "Source": "StackExchange",
                "Keyword": keyword,
                "Title": item.get("title"),
                "Content": item.get("link"),
                "URL": item.get("link"),
                "Date": None
            })
        df = pd.DataFrame(results)
    except Exception as e:
        logging.error(f"Error fetching StackExchange data: {e}")
        st.error(f"Error fetching StackExchange data: {e}")
    return df

@lru_cache(maxsize=128)
def fetch_mediastack_data(keyword, limit=20):
    logging.info(f"Fetching Mediastack data for '{keyword}'...")
    url = "http://api.mediastack.com/v1/news"
    params = {
        "access_key": MEDIASTACK_API_KEY,
        "keywords": keyword,
        "limit": limit,
        "languages": "en"
    }
    df = pd.DataFrame()
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        articles = data.get("data", [])
        results = []
        for art in articles:
            results.append({
                "Source": "Mediastack",
                "Keyword": keyword,
                "Title": art.get("title"),
                "Content": art.get("description"),
                "URL": art.get("url"),
                "Date": art.get("published_at")
            })
        df = pd.DataFrame(results)
    except Exception as e:
        logging.error(f"Error fetching Mediastack data: {e}")
        st.error(f"Error fetching Mediastack data: {e}")
    return df

@lru_cache(maxsize=128)
def fetch_google_trends_data(keywords_tuple, timeframe="today 12-m", geo=""):
    """
    We pass a tuple instead of a list so @lru_cache can handle it (tuples are hashable).
    """
    logging.info(f"Fetching Google Trends data for {keywords_tuple} with timeframe={timeframe}...")
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = list(keywords_tuple)  # convert tuple back to a list
    df = pd.DataFrame()
    try:
        pytrends.build_payload(kw_list=kw_list, timeframe=timeframe, geo=geo)
        raw_df = pytrends.interest_over_time()
        if 'isPartial' in raw_df.columns:
            raw_df = raw_df.drop(columns=['isPartial'])
        raw_df = raw_df.reset_index().melt(
            id_vars='date',
            value_vars=kw_list,
            var_name='Keyword',
            value_name='Interest'
        )
        raw_df['Source'] = "GoogleTrends"
        raw_df['URL'] = None
        raw_df['Content'] = None
        raw_df['Title'] = None
        raw_df['Date'] = raw_df['date'].astype(str)
        raw_df.drop(columns=['date'], inplace=True)
        df = raw_df
    except Exception as e:
        logging.error(f"Error fetching Google Trends data: {e}")
        st.error(f"Error fetching Google Trends data: {e}")
    return df

#########################
# SENTIMENT
#########################

def analyze_sentiment(df, text_column="Content"):
    if text_column not in df.columns:
        return df
    texts = df[text_column].fillna("")
    polarities = []
    subjectivities = []
    for t in texts:
        blob = TextBlob(t)
        polarities.append(blob.sentiment.polarity)
        subjectivities.append(blob.sentiment.subjectivity)
    df["Sentiment_Polarity"] = polarities
    df["Sentiment_Subjectivity"] = subjectivities
    return df

#########################
# GPT SUMMARIZATION
#########################

def summarize_with_gpt(texts, prompt):
    if not OPENAI_API_KEY:
        return "No OPENAI_API_KEY provided."
    combined_text = "\n".join(texts)
    # Limit to ~3000 chars to avoid hitting token limits
    combined_text = combined_text[:3000]
    full_prompt = f"{prompt}\n\nHere is the data:\n{combined_text}\n\nPlease respond accordingly."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in LLM request: {e}"

#########################
# STREAMLIT DASHBOARD
#########################

st.title("Comprehensive Market Research Dashboard with GPT Enhancement")
st.write("A robust tool integrating multiple data sources, sentiment analysis, GPT insights, and advanced charts.")

keyword_input = st.text_input("Enter keywords (comma-separated)", "manicure tools,pedicure tools")
keywords = [kw.strip() for kw in keyword_input.split(",") if kw.strip()]

st.subheader("Data Sources")
col1, col2 = st.columns(2)
with col1:
    include_reddit = st.checkbox("Reddit", True)
    include_youtube = st.checkbox("YouTube", True)
    include_news = st.checkbox("NewsAPI", True)
    include_google = st.checkbox("Google Custom Search", False)
    include_yelp = st.checkbox("Yelp (NYC)", False)
with col2:
    include_stackexchange = st.checkbox("StackExchange (Beauty)", False)
    include_mediastack = st.checkbox("Mediastack", False)
    include_trends = st.checkbox("Google Trends", True)
    apply_sentiment = st.checkbox("Apply Sentiment Analysis", True)

st.subheader("Analysis Settings")
sentiment_filter = st.slider("Minimum Sentiment Polarity (-1.0 to 1.0)", -1.0, 1.0, -1.0)
crosstab_metric = st.selectbox("Crosstab Metric", ["Source", "Keyword"], index=0)
timeframe = st.selectbox("Google Trends Timeframe", ["today 12-m", "today 5-y", "all"], index=0)

st.subheader("GPT Integration")
use_llm = st.checkbox("Use GPT to Summarize/Analyze")
user_prompt = ""
if use_llm:
    user_prompt = st.text_area("GPT Prompt", "Summarize the main findings from this data.")

if st.button("Run Query"):
    dataframes = []
    for kw in keywords:
        if include_reddit:
            df_r = fetch_reddit_data(kw)
            dataframes.append(df_r)
        if include_youtube:
            df_y = fetch_youtube_data(kw)
            dataframes.append(df_y)
        if include_news:
            df_n = fetch_news_data(kw)
            dataframes.append(df_n)
        if include_google:
            df_g = fetch_google_search_data(kw)
            dataframes.append(df_g)
        if include_yelp:
            df_yelp = fetch_yelp_data(kw)
            dataframes.append(df_yelp)
        if include_stackexchange:
            df_se = fetch_stackexchange_data(kw)
            dataframes.append(df_se)
        if include_mediastack:
            df_m = fetch_mediastack_data(kw)
            dataframes.append(df_m)

    if include_trends and len(keywords) > 0:
        # Convert list to tuple for caching
        dataframes.append(fetch_google_trends_data(tuple(keywords), timeframe=timeframe))

    if len(dataframes) > 0:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.drop_duplicates(subset=["URL", "Title", "Source", "Keyword"], inplace=True)

        if apply_sentiment:
            combined_df = analyze_sentiment(combined_df, "Content")
            combined_df = combined_df[combined_df["Sentiment_Polarity"] >= sentiment_filter]

        st.subheader("Combined Data")
        st.write("All data from selected sources:")
        st.dataframe(combined_df)

        if not combined_df.empty and "Source" in combined_df.columns:
            source_counts = combined_df["Source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Count"]
            fig_source = px.bar(source_counts, x="Source", y="Count", title="Count by Source")
            st.plotly_chart(fig_source)

        if apply_sentiment and not combined_df.empty and "Sentiment_Polarity" in combined_df.columns:
            nbins = st.slider("Bins for Sentiment Histogram", 5, 50, 20)
            fig_senti = px.histogram(combined_df, x="Sentiment_Polarity", nbins=nbins, title="Sentiment Polarity Distribution")
            st.plotly_chart(fig_senti)

        # Crosstab
        if not combined_df.empty and crosstab_metric in combined_df.columns and "Sentiment_Polarity" in combined_df.columns:
            bins = [-1, -0.1, 0.1, 1]
            labels = ["Negative", "Neutral", "Positive"]
            combined_df["Sentiment_Category"] = pd.cut(combined_df["Sentiment_Polarity"], bins=bins, labels=labels)
            ctab = pd.crosstab(combined_df[crosstab_metric], combined_df["Sentiment_Category"])
            st.subheader(f"Crosstab: {crosstab_metric} vs. Sentiment Category")
            st.dataframe(ctab)
            fig_ctab = px.bar(ctab, title=f"{crosstab_metric} vs. Sentiment Category", barmode="group")
            st.plotly_chart(fig_ctab)

        # If we have multiple keywords and used Google Trends
        if include_trends and len(keywords) > 1:
            trends_data = combined_df[combined_df["Source"] == "GoogleTrends"]
            if not trends_data.empty:
                interest_summary = trends_data.groupby("Keyword")["Interest"].mean().reset_index()
                fig_trends = px.bar(interest_summary, x="Keyword", y="Interest", title="Avg Interest by Keyword (Google Trends)")
                st.plotly_chart(fig_trends)

        st.subheader("Advanced & 3D Visualizations")
        viz_choice = st.selectbox("Visualization Type", ["3D Scatter (Sentiment)", "Scatter (Polarity vs Subjectivity)"])
        if apply_sentiment and not combined_df.empty and "Sentiment_Polarity" in combined_df.columns and "Sentiment_Subjectivity" in combined_df.columns:
            if "Interest" not in combined_df.columns:
                combined_df["Interest"] = 0
            else:
                combined_df["Interest"] = pd.to_numeric(combined_df["Interest"], errors="coerce").fillna(0)

            if viz_choice == "3D Scatter (Sentiment)":
                fig_3d = px.scatter_3d(
                    combined_df,
                    x="Sentiment_Polarity",
                    y="Sentiment_Subjectivity",
                    z="Interest",
                    color=crosstab_metric,
                    title="3D Scatter: Polarity vs. Subjectivity vs. Interest",
                    opacity=0.7
                )
                st.plotly_chart(fig_3d)
            else:
                fig_scatter = px.scatter(
                    combined_df,
                    x="Sentiment_Polarity",
                    y="Sentiment_Subjectivity",
                    color=crosstab_metric,
                    hover_data=["Title", "Source"],
                    title="Scatter: Polarity vs. Subjectivity"
                )
                st.plotly_chart(fig_scatter)

        # GPT Summaries
        if use_llm and user_prompt.strip():
            st.subheader("LLM (GPT) Analysis")
            # We only combine the first 5 Titles + 5 Contents to avoid token overflows
            snippet_texts = combined_df["Title"].fillna("").head(5).tolist() + combined_df["Content"].fillna("").head(5).tolist()
            gpt_response = summarize_with_gpt(snippet_texts, user_prompt)
            st.write("**GPT Response:**")
            st.write(gpt_response)

        # Download
        st.download_button("Download Combined CSV", combined_df.to_csv(index=False), "combined_data.csv")

    else:
        st.write("No data returned. Try enabling more sources or changing the keywords.")

st.write("If you want to share remotely, run ngrok in another terminal:")
st.code(
    "ngrok config add-authtoken 2qVTNHqFkTxL3lHy2k5JMDDmizV_9TCMxjyRWpa3zZjqwKV3\n"
    "streamlit run dashboard.py\n"
    "ngrok http 8501\n"
    "## Then share the forwarding URL shown by ngrok."
)
