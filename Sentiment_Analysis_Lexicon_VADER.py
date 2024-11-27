from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Define function to get sentiment score from text
def vader_score(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']


# Add sentiment label (positive or negative) based on compound VADER score
def sentiment_label(compound_score):
    if compound_score > 0.5:
        return 'positive'
    elif compound_score < -0.5:
        return 'negative'
    else:
        return 'neutral'


# Main function
def main(input_file):
    df = pd.read_csv(input_file)

    print("Excerpt of DF: " + df.head())

    print("Performing sentiment analysis...")

    df['vader_score'] = df['processed_reviews'].apply(vader_score)
    df['vader_label'] = df['vader_score'].apply(sentiment_label)

    output_file = 'Data/reviews_VADER.csv'
    df.to_csv(output_file, index=False)


main("Data/reviews_preprocessed.csv")
