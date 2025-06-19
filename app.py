import streamlit as st
import pandas as pd
from recommender import get_recommendations, df

st.title("📰 News Recommendation Engine")

# Dropdown to select a news article
article_titles = df['title'].tolist()
selected_title = st.selectbox("Choose a news article:", article_titles)

# Select sentiment filter (optional)
sentiment = st.selectbox("Filter by sentiment (optional):", [
                         'None', 'positive', 'negative', 'neutral'])

# Convert None string to actual None
sentiment_filter = None if sentiment == 'None' else sentiment

# Find index
index_to_search = df[df['title'] == selected_title].index[0]

if st.button("Get Recommendations"):
    results = get_recommendations(
        index_to_search, top_n=5, sentiment_filter=sentiment_filter)
    st.subheader("🔍 Top Recommendations:")
    for title, sentiment, score in results:
        st.markdown(f"**{title}** ({sentiment}) — Score: {score:.2f}")
