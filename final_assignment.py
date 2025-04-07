"""
Author: Emanuele Bossi
Class: DS 244
Date: 2025-04-16
Presentation Date: 2025-04-22
Description: Sentiment analyzer for TMDb film reviews.
Professor: Wolf Paulus
"""

import os
import json
import requests
import mysql.connector
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st

# Ensure VADER lexicon is downloaded
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# TMDb API Key (left here for reference — should be stored securely)
API_KEY = os.getenv("TMDB_API_KEY", "4ceb56a4ccbbd21b6f73e6cc6177d5be")

# Database config
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "user2",
    "password": "erau2025",
    "database": "db2"
}

def extract_reviews(movie_id: int, output_file: str = "movie_reviews.json") -> None:
    """Fetches reviews from TMDb API and saves to JSON."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status()
        reviews = [{
            "movie_id": movie_id,
            "author": r.get("author"),
            "rating": r.get("author_details", {}).get("rating"),
            "content": r.get("content"),
            "created": r.get("created_at")
        } for r in response.json().get("results", [])]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(reviews, f, indent=4, ensure_ascii=False)
    except requests.RequestException as e:
        st.error(f"Error fetching reviews: {e}")

def clean_text(text: str) -> str:
    """Cleans review text by removing HTML tags."""
    return BeautifulSoup(text, "html.parser").get_text().strip()

def parse_review_date(iso_date: str) -> Optional[str]:
    """Converts ISO 8601 date string to MySQL DATETIME format."""
    try:
        iso_date = iso_date.split('.')[0].rstrip('Z')
        dt = datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

def insert_reviews_into_db(file_name: str, movie_id: Optional[int] = None) -> List[dict]:
    """Reads reviews from JSON and inserts them into MySQL."""
    with open(file_name, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS movie_reviews (
                       id INT AUTO_INCREMENT PRIMARY KEY,
                       movie_id INT,
                       author VARCHAR(255),
                       content TEXT,
                       rating FLOAT,
                       review_date DATETIME
                    )
                """)
        
        # Clear previous data
        cursor.execute("DELETE FROM movie_reviews")
        
        for review in reviews:
            mid = review.get("movie_id") if movie_id is None else movie_id
            author = review.get("author", "Unknown")
            content = clean_text(review.get("content", ""))
            rating = review.get("rating")
            review_date = parse_review_date(review.get("created", ""))

            cursor.execute("""
                INSERT INTO movie_reviews (movie_id, author, content, rating, review_date)
                VALUES (%s, %s, %s, %s, %s)
            """, (mid, author, content, rating, review_date))


        conn.commit()
        return reviews
    except mysql.connector.Error as e:
        st.error(f"MySQL Error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def fetch_reviews_from_db() -> pd.DataFrame:
    """Fetches all reviews from the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT movie_id, author, content, rating, review_date FROM movie_reviews")
        results = cursor.fetchall()
        return pd.DataFrame(results)
    except mysql.connector.Error as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()
        conn.close()

def classify_sentiment(text: str) -> str:
    """Classifies sentiment using VADER."""
    score = sia.polarity_scores(text)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    return "Neutral"



# Streamlit UI
st.title("🎬 TMDb Movie Review Sentiment Analyzer")

# Extract reviews for 10 different films
# The Shawshank Redemption: 278
# The Godfather: 238
# The Godfather Part II: 240
# Pulp Fiction: 680
# The Good, the Bad and the Ugly: 425
# Angry Men: 479
# The Dark Knight: 155
# Schindler's List: 424
# The Lord of the Rings: The Return of the King: 122
# Fight Club: 550
movie_ids = [278, 238, 240, 680, 425, 479, 155, 424, 122, 550]

all_reviews = []
for movie_id in movie_ids:
    extract_reviews(movie_id, output_file=f"temp_reviews_{movie_id}.json")
    with open(f"temp_reviews_{movie_id}.json", "r", encoding="utf-8") as f:
        all_reviews.extend(json.load(f))

# Save combined reviews
with open("all_movie_reviews.json", "w", encoding="utf-8") as f:
    json.dump(all_reviews, f, indent=4, ensure_ascii=False)

# Insert reviews into the MySQL database
reviews = insert_reviews_into_db("all_movie_reviews.json", movie_id=None)

# Fetch reviews from the database and display them
df = fetch_reviews_from_db()
if df.empty:
    st.warning("No reviews found in the database.")
    st.stop()

df["review_date"] = pd.to_datetime(df["review_date"])
df["sentiment"] = df["content"].apply(classify_sentiment)

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
start_date = st.sidebar.date_input("Start Date", df["review_date"].min().date())
end_date = st.sidebar.date_input("End Date", df["review_date"].max().date())

filtered_df = df[
    (df["review_date"].dt.date >= start_date) &
    (df["review_date"].dt.date <= end_date)
]

movie_id_options = df["movie_id"].unique()
selected_movie_id = st.sidebar.selectbox("Select Movie ID", options=movie_id_options)
filtered_df = filtered_df[filtered_df["movie_id"] == selected_movie_id]

keyword = st.sidebar.text_input("Search Reviews")
if keyword:
    filtered_df = filtered_df[filtered_df["author"].str.contains(keyword, case=False, na=False)]

st.write(f"Displaying {len(filtered_df)} reviews from {start_date} to {end_date}")
st.dataframe(filtered_df[["author", "content", "rating", "sentiment", "review_date"]])

# Sentiment distribution
st.subheader("📊 Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(x="sentiment", data=filtered_df, palette="coolwarm", ax=ax)
st.pyplot(fig)

# Sentiment trend over time
st.subheader("📈 Sentiment Trends Over Time")
filtered_df["date"] = filtered_df["review_date"].dt.date
trends = filtered_df.groupby(["date", "sentiment"]).size().unstack().fillna(0)

fig2, ax2 = plt.subplots(figsize=(10, 5))
trends.plot(marker="o", ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# WordCloud visualization
st.subheader("☁️ Word Cloud of Review Content")

# Join all review content into a single string
all_text = " ".join(filtered_df["content"].dropna())
# Generate word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="coolwarm",
    stopwords=set(WordCloud().stopwords).union({"movie", "film", "one"})
).generate(all_text)

# Display the WordCloud
fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)