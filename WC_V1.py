import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter


def create_word_cloud(file_path, n_words):
    # Read CSV File
    df = pd.read_csv(file_path, header=None)

    # Combine all rows into a single string
    text_input = ' '.join(df[0].astype(str))

    print("First few rows of the data:")
    print(df.head())  # To verify data loaded correctly

    # Count word frequencies
    word_counts = Counter(text_input.split())

    # Get the top N most common words
    top_n_words = dict(word_counts.most_common(n_words))

    print(f"\nTop {n_words} words:")
    for word, count in top_n_words.items():
        print(f"{word}: {count}")

    # Generate Word Cloud
    wordcloud = WordCloud(width=800, height=600, background_color='white').generate_from_frequencies(top_n_words)

    # Display the word cloud
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# Create word cloud
create_word_cloud("Data/reviews_preprocessed.csv", 100)
