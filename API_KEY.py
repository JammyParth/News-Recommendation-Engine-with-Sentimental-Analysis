import requests
import pandas as pd

API_KEY = '42ea5cdda92845499ddb7d7769e71f9a'
URL = 'https://newsapi.org/v2/top-headlines'
PARAMS = {
    'apiKey': API_KEY,
    'country': 'us',
    'category': 'sports',  # change to 'sports', 'business', etc.
    'pageSize': 50
}

response = requests.get(URL, params=PARAMS)
data = response.json()

articles = data.get('articles', [])

# Convert to DataFrame
df = pd.DataFrame([{
    'title': article['title'],
    'description': article['description'],
    'content': article['content'],
    'url': article['url'],
    'publishedAt': article['publishedAt']
} for article in articles])

# Save to CSV
df.to_csv('news_data.csv', index=False)
print("✅ News data saved to 'news_data.csv'")
