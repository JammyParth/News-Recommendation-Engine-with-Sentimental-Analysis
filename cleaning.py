
from textblob import TextBlob
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Load data
df = pd.read_csv('news_data.csv')

# Combine title + description for analysis
df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

# Preprocess function


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


df['clean_text'] = df['text'].apply(clean_text)


# 2nd step


# Sentiment function

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity  # Range: [-1.0, 1.0]


df['sentiment'] = df['clean_text'].apply(get_sentiment)

# Optional: Label the sentiment


def label_sentiment(score):
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'


df['sentiment_label'] = df['sentiment'].apply(label_sentiment)

# Save this processed data
df.to_csv('news_data_cleaned.csv', index=False)
print("✅ Sentiment analysis complete. Saved as 'news_data_cleaned.csv'")
