from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')       # Correct tokenizers
nltk.download('punkt_tab')   # For tab tokenization
nltk.download('stopwords')    # For filtering
text = "This is a test."
print(word_tokenize(text))
