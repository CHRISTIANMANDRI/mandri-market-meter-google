import streamlit as st
import requests
import pandas as pd
import subprocess
import json
import time
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt

# Load FinBERT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_model()

# Sentiment scoring helper
def sentiment_score(label):
    return {"positive": 1, "neutral": 0, "negative": -1}.get(label, 0)

# Enhanced tweet search
def fetch_tweets(ticker, count=10):
    query = f'${ticker} OR "{ticker}" lang:en'
    command = f"snscrape --max-results {count} --jsonl twitter-search '{query}'"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    tweets = []
    for line in result.stdout.splitlines():
        try:
            tweet = json.loads(line)
            tweets.append(tweet["content"])
        except:
            continue
    return tweets

# Google News search (scrape headlines)
def fetch_google_news(ticker, count=10):
    url = f"https://news.google.com/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    articles = soup.select("article h3")
    headlines = [a.text.strip() for a in articles][:count]
    return headlines

# Google Trends
def fetch_trend_score(keyword):
    pytrends = TrendReq()
    pytrends.build_payload([keyword], timeframe="today 3-m")
    df = pytrends.interest_over_time()
    if not df.empty:
        recent_avg = df[keyword].tail(7).mean()
        return round(recent_avg, 2)
    return 0

# Investment signal logic
def get_signal(score):
    if score > 0.4:
        return "BUY"
    elif score < -0.4:
        return "SELL"
    return "HOLD"

# UI
st.title("📈 Mandri Market Meter with Google Search Intelligence")

ticker = st.text_input("Enter Ticker (e.g., TSLA, AAPL, MSFT)", "TSLA").upper()

with st.spinner("Fetching and analyzing data..."):
    tweets = fetch_tweets(ticker)
    gnews = fetch_google_news(ticker)
    trend_score = fetch_trend_score(ticker)

    all_texts = tweets + gnews
    if all_texts:
        sentiment_results = sentiment_pipeline(all_texts)
        scores = [sentiment_score(r["label"]) for r in sentiment_results]
        avg_sentiment = sum(scores) / len(scores)
        combined_sentiment = avg_sentiment * (1 + trend_score / 100)  # Weight by search interest
    else:
        avg_sentiment = 0
        combined_sentiment = 0

signal = get_signal(combined_sentiment)

# Display results
st.metric("Twitter + Google News Sentiment", f"{avg_sentiment:.2f}")
st.metric("Google Trends Score", f"{trend_score:.2f}")
st.metric("Weighted Sentiment", f"{combined_sentiment:.2f}")
st.subheader(f"💡 Investment Signal: **{signal}**")

fig, ax = plt.subplots(figsize=(5, 1.5))
ax.barh(0, combined_sentiment, color="green" if combined_sentiment > 0 else "red" if combined_sentiment < 0 else "gray")
ax.set_xlim([-1, 1])
ax.set_yticks([])
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_title("Combined Sentiment Gauge")
st.pyplot(fig)

if st.checkbox("Show Headlines and Tweets with Sentiment"):
    for t, r in zip(all_texts, sentiment_results):
        st.write(f"**{t}** — `{r['label']}` ({r['score']:.2f})")
