import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# load cleaned data
df = pd.read_csv('news_data_cleaned.csv')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to get sentiment scores


def analyze_sentiment(text):
    score = sid.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


# Apply sentiment analysis
df['sentiment'] = df['clean_text'].apply(analyze_sentiment)

df.to_csv('news_data_with_sentiment.csv', index=False)
print("✅ Sentiment analysis complete. Saved as 'news_data_with_sentiment.csv'")
