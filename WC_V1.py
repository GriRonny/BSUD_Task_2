import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter


def create_word_clouds(file_path, n_words, custom_stopwords=None):
    # Read CSV File
    df_processed = pd.read_csv(file_path)

    # Ensure column names are properly set
    if "Tokens" not in df_processed.columns or "Label" not in df_processed.columns:
        raise ValueError("CSV file must have 'Tokens' and 'Label' columns.")

    # Default stopwords
    if custom_stopwords is None:
        custom_stopwords = set()
    else:
        custom_stopwords = set(custom_stopwords)

    # Filter by sentiment
    positive_reviews = df_processed[df_processed["Label"] == 'positive']["Tokens"]
    neutral_reviews = df_processed[df_processed["Label"] == 'neutral']["Tokens"]
    negative_reviews = df_processed[df_processed["Label"] == 'negative']["Tokens"]

    # Function to generate and display a word cloud
    def generate_word_cloud(reviews, sentiment):
        # Combine all reviews into a single string
        text_input = ' '.join(reviews.astype(str))

        # Filter custom stopwords
        word_counts = Counter(text_input.split())
        filtered_word_counts = {word: count for word, count in word_counts.items() if
                                word.lower() not in custom_stopwords}

        # Get the top N most common words
        top_n_words = dict(Counter(filtered_word_counts).most_common(n_words))

        # Generate Word Cloud
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white'
        ).generate_from_frequencies(top_n_words)

        # Display the Word Cloud
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"{sentiment.capitalize()} Reviews Word Cloud", fontsize=18)
        plt.show()

    # Generate word clouds for each sentiment
    print("\nGenerating word cloud for positive reviews...")
    generate_word_cloud(positive_reviews, "positive")

    print("\nGenerating word cloud for neutral reviews...")
    generate_word_cloud(neutral_reviews, "neutral")

    print("\nGenerating word cloud for negative reviews...")
    generate_word_cloud(negative_reviews, "negative")


# Define custom stopwords
custom_stopwords = {"app", "spotify", "play", "song", "music", "use"}

# Create word clouds
create_word_clouds("Data/reviews_preprocessed.csv", 100, custom_stopwords)
