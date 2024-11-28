from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Define function to get sentiment score from text
def vader_score(text):
    """Calculate VADER compound sentiment score."""
    scores = analyzer.polarity_scores(text)
    return scores['compound']


# Add sentiment label (positive or negative) based on compound VADER score
def sentiment_label(compound_score, positive_threshold=0.5, negative_threshold=-0.5):
    """Assign sentiment label based on thresholds."""
    if compound_score > positive_threshold:
        return 'positive'
    elif compound_score < negative_threshold:
        return 'negative'
    else:
        return 'neutral'


# Perform sentiment analysis (optimized version)
def vader_analysis(text):
    """Return both VADER compound score and sentiment label."""
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    label = sentiment_label(compound_score)
    return pd.Series([compound_score, label], index=['vader_score', 'vader_label'])


# Visualization function
def visualize_sentiment_distribution(df):
    """Create a bar plot showing the distribution of sentiment labels."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sentiment_counts = df['vader_label'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Count")
    plt.show()


# Main function
def main(input_file):
    """Perform sentiment analysis on processed reviews in a CSV file."""
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file '{input_file}' is empty or invalid.")

    if 'processed_reviews' not in df.columns:
        raise KeyError("The required column 'processed_reviews' is missing from the input file.")

    logging.info("Performing sentiment analysis...")
    df[['vader_score', 'vader_label']] = df['processed_reviews'].apply(vader_analysis)

    # Save results to CSV
    output_file = 'Data/reviews_VADER.csv'
    df.to_csv(output_file, index=False)
    logging.info(f"Sentiment analysis complete. Results saved to {output_file}.")

    # Visualization
    logging.info("Creating sentiment distribution visualization...")
    visualize_sentiment_distribution(df)


# Call the main function
if __name__ == "__main__":
    main("Data/reviews_preprocessed.csv")
