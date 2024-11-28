import string
import logging
import pandas as pd
import spacy
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re

# Setup logging
logging.basicConfig(level=logging.INFO)

# Download resources
logging.info("Downloading NLTK resources...")
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model
logging.info("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')

# Load data
df = pd.read_csv('Data/spotify_reviews_lightweight.csv', header=None, names=['raw_reviews'])


# Preprocessing function
def preprocess_text(text):
    """Preprocess a single text string."""
    # Lowercase
    text = text.lower()

    # Remove URLs and emails
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation, non-alphanumeric tokens, and stopwords
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words and t not in string.punctuation]

    # Lemmatization (only alpha tokens)
    lemmatized_tokens = [token.lemma_ for token in nlp(' '.join(tokens)) if token.is_alpha]

    return ' '.join(lemmatized_tokens)


# Apply preprocessing with progress bar
logging.info("Preprocessing started...")
tqdm.pandas(desc="Preprocessing Reviews")
# Create column header for processed reviews and populate with preprocessed text
df['processed_reviews'] = df['raw_reviews'].progress_apply(preprocess_text)
logging.info("Preprocessing finished...")

# Save to CSV
output_file = 'Data/reviews_preprocessed.csv'
df['processed_reviews'].to_csv(output_file, index=False)

# Print unprocessed and processed text
print("Unprocessed Text (first few words):")
print(df['raw_reviews'].iloc[0][:50])

print("\nProcessed Text (first few words):")
print(df['processed_reviews'].iloc[0][:50])
