import string
import logging
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import mark_negation

# Based on https://www.geeksforgeeks.org/removing-stop-words-nltk-python/ and chapter 7/8

# Setup logging
logging.basicConfig(level=logging.INFO)

# Download resources
logging.info("Downloading Resources...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# spaCy model for lemmatization (medium model) "python -m spacy download en_core_web_sm"
logging.info("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')

# Load data
df = pd.read_csv('Data/spotify_reviews_lightweight.csv', header=None, names=['raw_reviews'])


# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Mark negations
    tokens = mark_negation(tokens)

    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]

    # Lemmatization
    lemmatized_tokens = [token.lemma_ for token in nlp(' '.join(tokens))]

    return ' '.join(lemmatized_tokens)


# Apply preprocessing
logging.info("Preprocessing started...")
df['processed_reviews'] = df['raw_reviews'].apply(preprocess_text)
logging.info("Preprocessing finished...")

# Save to CSV (into Data folder)
output_file = 'Data/reviews_preprocessed.csv'
df['processed_reviews'].to_csv(output_file, index=False)

# Print first few words of unprocessed and processed text
print("Unprocessed Text (first few words):")
print(df['raw_reviews'].iloc[0][:50])

print("\nProcessed Text (first few words):")
print(df['processed_reviews'].iloc[0][:50])
