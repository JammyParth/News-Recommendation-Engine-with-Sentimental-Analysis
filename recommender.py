import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('news_data_with_sentiment.csv')

# Fill missing values
for col in ['title', 'description', 'clean_text']:
    if col in df.columns:
        df[col] = df[col].fillna('')
    else:
        df[col] = ''

# Combine relevant fields for better semantic richness
df['combined'] = df['title'] + ' ' + df['description'] + ' ' + df['clean_text']

# Ensure non-empty text
df = df[df['combined'].str.strip() != '']

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommender


def get_recommendations(index, top_n=5, sentiment_filter=None):
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
        1:top_n+1]

    recommendations = []
    for i, score in sim_scores:
        if sentiment_filter:
            if df.iloc[i]['sentiment'] == sentiment_filter:
                recommendations.append(
                    (df.iloc[i]['title'], df.iloc[i]['sentiment'], score))
        else:
            recommendations.append(
                (df.iloc[i]['title'], df.iloc[i]['sentiment'], score))
    return recommendations


# new query to search by user input:
def search_by_query(query, top_n=5, sentiment_filter=None):
    # Vectorize the query using the same TF-IDF vectorizer
    query_vec = vectorizer.transform([query])

    # Compute cosine similarity with all articles
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top N scores
    sim_indices = sim_scores.argsort()[::-1][:top_n]

    recommendations = []
    for i in sim_indices:
        if sentiment_filter:
            if df.iloc[i]['sentiment'] == sentiment_filter:
                recommendations.append(
                    (df.iloc[i]['title'], df.iloc[i]['sentiment'], sim_scores[i]))
        else:
            recommendations.append(
                (df.iloc[i]['title'], df.iloc[i]['sentiment'], sim_scores[i]))

    return recommendations


if __name__ == "__main__":
    print("🔍 Search by Query Mode")
    user_query = input("Enter your search query: ")
    sentiment = input(
        "Filter by sentiment? (positive/negative/neutral/none): ").strip().lower()

    if sentiment == "none" or sentiment == "":
        sentiment = None

    results = search_by_query(user_query, top_n=5, sentiment_filter=sentiment)

    print(f"\nTop recommendations for query: \"{user_query}\"")
    for title, sentiment, score in results:
        print(f"- {title} ({sentiment}) [Score: {score:.2f}]")
