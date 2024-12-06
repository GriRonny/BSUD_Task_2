import string
import logging
import pandas as pd
import spacy
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re  # Regex package
import emoji

# Setup logging
logging.basicConfig(level=logging.INFO)

# Download NLTK resources
logging.info("Downloading NLTK resources...")
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model
logging.info("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')

# Load data
df = pd.read_csv('../Data/spotify_reviews_lightweight.csv', header=None, names=['raw_reviews'])

# Function to map emojis to text. E.g., "Python is 👍" is transformed to "Python is :thumbs_up:"
def map_emojis(text):
    """Convert emojis to their text representations."""
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Replace underscores with spaces in emoji descriptions to avoid them being omitted during tokenization.
    return text.replace('_', ' ')

# Preprocessing function
def preprocess_text(text):
    """Preprocess a single text string."""
    # Lowercase
    text = text.lower()

    # Convert emojis to text
    text = map_emojis(text)

    # Remove URLs and emails
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Tokenization and Lemmatization using spaCy
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_punct and not token.is_stop
    ]

    return ' '.join(tokens)

# Apply preprocessing with progress bar
logging.info("Preprocessing started...")
tqdm.pandas(desc="Preprocessing Reviews")
df['processed_reviews'] = df['raw_reviews'].progress_apply(preprocess_text)
logging.info("Preprocessing finished...")

# Save to CSV
output_file = '../Data/reviews_preprocessed.csv'
df['processed_reviews'].to_csv(output_file, index=False)

# Print unprocessed and processed text for verification
print("Unprocessed Text (first few words):")
print(df['raw_reviews'].iloc[1][:50])

print("\nProcessed Text (first few words):")
print(df['processed_reviews'].iloc[1][:50])
