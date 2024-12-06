from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Add sentiment label based on VADER's default thresholds
def sentiment_label(compound_score, positive_threshold=0.05, negative_threshold=-0.05):
    """Assign sentiment label based on thresholds."""
    if compound_score >= positive_threshold:
        return 'positive'
    elif compound_score <= negative_threshold:
        return 'negative'
    else:
        return 'neutral'


# Perform sentiment analysis
def vader_analysis(text):
    """Return both VADER compound score and sentiment label."""
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    label = sentiment_label(compound_score)
    return pd.Series([compound_score, label], index=['vader_score', 'vader_label'])


# Visualization functions
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

def visualize_compound_score_distribution(df):
    """Create a histogram showing the distribution of compound sentiment scores."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['vader_score'], bins=30, kde=True)
    plt.title("Compound Score Distribution")
    plt.xlabel("VADER Compound Score")
    plt.ylabel("Frequency")
    plt.show()


# Main function
def main(input_file):
    """Perform sentiment analysis on raw reviews in a CSV file."""
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file '{input_file}' is empty or invalid.")

    if 'raw_reviews' not in df.columns:
        raise KeyError("The required column 'raw_reviews' is missing from the input file.")

    logging.info("Performing sentiment analysis...")
    tqdm.pandas(desc="Apply Vader Analysis")
    df[['vader_score', 'vader_label']] = df['raw_reviews'].progress_apply(vader_analysis)

    # Save results to CSV
    output_file = 'Data/VADER/reviews_VADER.csv'
    df.to_csv(output_file, index=False)
    logging.info(f"Sentiment analysis complete. Results saved to {output_file}.")

    # Visualization
    logging.info("Creating sentiment distribution visualization...")
    visualize_sentiment_distribution(df)
    visualize_compound_score_distribution(df)


# Call the main function
if __name__ == "__main__":
    main("Data/spotify_reviews_lightweight.csv")
