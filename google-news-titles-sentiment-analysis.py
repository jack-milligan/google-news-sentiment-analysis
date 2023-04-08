"""
This script fetches article titles from a given URL, filters them based on specified keywords, and performs
sentiment analysis using both the NRC Emotion Lexicon and the DistilBERT model. The results are combined in a pandas
DataFrame and saved to an Excel file. Additionally, the script generates a bar chart of the sentiment scores per
title for visualization.

Dependencies:

    pandas
    nltk
    requests
    BeautifulSoup
    matplotlib
    transformers
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import requests as requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from transformers import pipeline


def contains_keywords(item_searched, items_searched_for):
    """
    Check if a given string contains any of the provided keywords.

    Args:
        item_searched (str): The string to search for keywords in.
        items_searched_for (list): A list of keywords to search for in the given string.

    Returns:
        bool: True if the given string contains any of the provided keywords, False otherwise.
    """
    contains_keyword = any(keyword in item_searched for keyword in items_searched_for)
    return contains_keyword


def get_title(article, tag):
    """
    Get the text of the specified tag in the given BeautifulSoup article object and convert it to lowercase.

    Args:
        article (bs4.element.Tag): A BeautifulSoup article object.
        tag (str): The tag to search for in the article object.

    Returns:
        str: The text of the specified tag in the given article object, converted to lowercase.
    """
    article_title = article.find(tag).text
    return article_title.lower()


def process_nrc_lexicon_line(line):
    """
    Process a line from the NRC Emotion Lexicon file, extracting the word, emotion, and score.

    Args:
        line (str): A line from the NRC Emotion Lexicon file.

    Returns:
        tuple: A tuple containing the word (str), emotion (str), and score (int).
    """
    word, emotion, score = line.strip().split('\t')
    score = int(score)
    return word, emotion, score


def add_to_nrc_lexicon(lexicon, word, emotion, score):
    """
    Add the given word, emotion, and score to the NRC lexicon dictionary.

    Args:
        lexicon (dict): The NRC Emotion Lexicon dictionary to add the data to.
        word (str): The word associated with the emotion and score.
        emotion (str): The emotion associated with the word and score.
        score (int): The score associated with the word and emotion.
    """
    if word not in lexicon:
        lexicon[word] = {emotion: score}
    else:
        lexicon[word][emotion] = score


def load_nrc_lexicon(file_path):
    """
    Load the NRC Emotion Lexicon from the specified file path.

    Args:
        file_path (str): The path to the NRC Emotion Lexicon file.

    Returns:
        dict: A dictionary containing the NRC Emotion Lexicon data.
    """
    nrc_lexicon = {}
    with open(file_path, 'r') as file:
        for line in file:
            word, emotion, score = process_nrc_lexicon_line(line)
            add_to_nrc_lexicon(nrc_lexicon, word, emotion, score)
    return nrc_lexicon


def tokenize_titles(article_titles):
    """
    Tokenize the given article titles.

    Args:
        article_titles (list): A list of article titles.

    Returns:
        list: A list of tokenized article titles.
    """
    tokenized_titles = [word_tokenize(title.lower()) for title in article_titles]
    return tokenized_titles


def analyze_title_nrc_sentiment(text_tokens, lexicon):
    """
    Analyze the sentiment of a tokenized title using the NRC Emotion Lexicon.

    Args:
        text_tokens (list): A list of tokens for a title.
        lexicon (dict): The NRC Emotion Lexicon dictionary.

    Returns:
        dict: A dictionary containing the sentiment scores for the title.
    """
    emotions = list(next(iter(lexicon.values())).keys())
    title_sentiment = {emotion: 0 for emotion in emotions}

    for token in text_tokens:
        if token in lexicon:
            token_sentiments = lexicon[token]
            for emotion, score in token_sentiments.items():
                title_sentiment[emotion] += score

    return title_sentiment


def analyze_nrc_sentiment(text, lexicon):
    """
    Analyze the sentiment of a list of titles using the NRC Emotion Lexicon.

    Args:
        text (list): A list of titles.
        lexicon (dict): The NRC Emotion Lexicon dictionary.

    Returns:
        list: A list of dictionaries containing the sentiment scores for each title.
    """
    tokenized_titles = tokenize_titles(text)
    sentiment_scores = [analyze_title_nrc_sentiment(title_tokens, lexicon) for title_tokens in tokenized_titles]
    return sentiment_scores


def get_sentiment_pipeline():
    """
    Get the DistilBERT sentiment analysis pipeline from the transformers library.

    Returns:
        transformers.Pipeline: The DistilBERT sentiment analysis pipeline.
    """
    return pipeline("sentiment-analysis")


def analyze_sentiment_using_distilbert(sentences, sentiment_pipeline):
    """
    Analyze the sentiment of a list of sentences using the DistilBERT sentiment analysis pipeline.

    Args:
        sentences (list): A list of sentences to analyze.
        sentiment_pipeline (transformers.Pipeline): The DistilBERT sentiment analysis pipeline.

    Returns:
        list: A list of dictionaries containing the sentiment analysis results for each sentence.
    """
    sentiment_scores = sentiment_pipeline(sentences)
    return sentiment_scores


def distilbert_sentiment_scores_to_dataframe(sentences, sentiment_scores):
    """
    Convert the DistilBERT sentiment scores to a pandas DataFrame.

    Args:
        sentences (list): A list of sentences that were analyzed.
        sentiment_scores (list): A list of dictionaries containing the sentiment analysis results.

    Returns:
        pd.DataFrame: A DataFrame containing the sentiment scores for each sentence.
    """
    data = []
    for sentence, sentiment_data in zip(sentences, sentiment_scores):
        label = sentiment_data['label'].lower().replace('positive', 'pos').replace('negative', 'neg')
        score = sentiment_data['score']
        data.append({"text": sentence, "sentiment_label": label, "sentiment_score": score})

    return pd.DataFrame(data)


def combined_analysis(article_titles, nrc_lexicon, sentiment_pipeline):
    """
    Perform combined NRC sentiment analysis and DistilBERT sentiment analysis on the given article titles.

    Args:
        article_titles (list): A list of article titles.
        nrc_lexicon (dict): The NRC Emotion Lexicon.
        sentiment_pipeline (transformers.Pipeline): The DistilBERT sentiment analysis pipeline.

    Returns:
        pd.DataFrame: A DataFrame containing the combined results of the NRC sentiment analysis and DistilBERT
    sentiment analysis.
    """
    # Perform NRC sentiment analysis
    nrc_sentiment_scores = analyze_nrc_sentiment(article_titles, nrc_lexicon)

    # Perform DistilBERT sentiment analysis
    distilbert_sentiment_scores = analyze_sentiment_using_distilbert(article_titles, sentiment_pipeline)

    # Combine NRC sentiment scores and DistilBERT sentiment scores in a DataFrame
    combined_data = []
    for title, nrc_scores, distilbert_data in zip(article_titles, nrc_sentiment_scores, distilbert_sentiment_scores):
        row_data = {
            "title": title,
            "distilbert_label": distilbert_data['label'].lower(),
            "distilbert_score": distilbert_data['score'],
        }
        row_data.update(nrc_scores)
        combined_data.append(row_data)

    return pd.DataFrame(combined_data)


def plot_sentiment(sentiment_scores, article_titles):
    """
    Plot the sentiment scores for a list of article titles using a horizontal bar chart.

    Args:
        sentiment_scores (list): A list of dictionaries containing the sentiment scores for each title.
        article_titles (list): A list of article titles.

    """
    emotions = list(sentiment_scores[0].keys())
    n_titles = len(article_titles)
    n_emotions = len(emotions)
    bar_width = 1 / (n_emotions + 1)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, emotion in enumerate(emotions):
        emotion_scores = [title_sentiment[emotion] for title_sentiment in sentiment_scores]
        ax.barh([j + i * bar_width for j in range(n_titles)], emotion_scores, height=bar_width, label=emotion)

    ax.set_yticks([j + (n_emotions / 2) * bar_width for j in range(n_titles)])
    ax.set_yticklabels(article_titles)
    ax.legend()

    plt.title("Sentiment Scores per Title")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Article Titles")
    plt.tight_layout()
    plt.show()


def fetch_titles(url, key_words, limit):
    """
    Fetches article titles from the given URL that contain the specified keywords.

    Args:
    url: str, the URL of the website to fetch articles from.
    key_words: list of str, keywords to filter the articles.
    limit: int, maximum number of articles to fetch.

    Returns:
        list of str, filtered article titles.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    nltk.download('punkt')

    titles = []

    for articles in soup.find_all('article'):
        try:
            title = get_title(articles, 'h4')
            if contains_keywords(title, key_words):
                titles.append(title)
        except AttributeError:
            continue

        if len(titles) == limit:
            break

    return titles


def perform_combined_sentiment_analysis(titles, nrc_lexicon, sentiment_pipeline):
    """
    Performs combined NRC and DistilBERT sentiment analysis on the given article titles.

    Args:
    titles: list of str, article titles.
    nrc_lexicon: dict, NRC emotion lexicon.
    sentiment_pipeline: Pipeline, DistilBERT sentiment analysis pipeline.

    Returns:
        DataFrame, combined sentiment analysis results.
    """
    combined_df = combined_analysis(titles, nrc_lexicon, sentiment_pipeline)
    return combined_df


def save_results_to_excel(combined_df, output_file):
    """
    Saves the combined sentiment analysis results to an Excel file.

    Args:
    combined_df: DataFrame, combined sentiment analysis results.
    output_file: str, file path for the output Excel file.
    """
    combined_df.to_excel(output_file, index=False)


def visualize_sentiment_analysis(titles, nrc_lexicon):
    """
    Visualizes the NRC sentiment analysis results using a bar chart.

    Args:
    titles: list of str, article titles.
    nrc_lexicon: dict, NRC emotion lexicon.
    """
    sentiment_scores = analyze_nrc_sentiment(titles, nrc_lexicon)
    plot_sentiment(sentiment_scores, titles)


def main(url, key_words, limit, nrc_lexicon_file, output_file):
    """
    Main function that orchestrates fetching article titles, performing sentiment analysis,
    saving the results, and visualizing the sentiment analysis.

    Args:
    url: str, the URL of the website to fetch articles from.
    key_words: list of str, keywords to filter the articles.
    limit: int, maximum number of articles to fetch.
    nrc_lexicon_file: str, file path for the NRC emotion lexicon file.
    output_file: str, file path for the output Excel file.
    """
    titles = fetch_titles(url, key_words, limit)
    nrc_lexicon = load_nrc_lexicon(nrc_lexicon_file)
    sentiment_pipeline = get_sentiment_pipeline()

    combined_df = perform_combined_sentiment_analysis(titles, nrc_lexicon, sentiment_pipeline)

    save_results_to_excel(combined_df, output_file)

    visualize_sentiment_analysis(titles, nrc_lexicon)


if __name__ == "__main__":
    url = 'https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB/sections' \
          '/CAQiQ0NCQVNMQW9JTDIwdk1EZGpNWFlTQW1WdUdnSlZVeUlOQ0FRYUNRb0hMMjB2TUcxcmVpb0pFZ2N2YlM4d2JXdDZLQUEqKggAKiYICiIgQ0JBU0Vnb0lMMjB2TURkak1YWVNBbVZ1R2dKVlV5Z0FQAVAB?hl=en-US&gl=US&ceid=US%3Aen'
    key_words = ['chatgpt', 'diffusion models']
    limit = 20
    nrc_lexicon_file = '/Users/jackmilligan/PycharmProjects/sentiment-analysis/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    output_file = 'sentiment_analysis.xlsx'

    main(url, key_words, limit, nrc_lexicon_file, output_file)
