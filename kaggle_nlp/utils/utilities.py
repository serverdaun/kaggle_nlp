import re
import pandas as pd


class PreprocessingUtils:

    @staticmethod
    def clean_keywords(tweets: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the keywords in the tweets DataFrame.
        Arguments:
            :tweets: DataFrame containing tweets with a 'keyword' column.
        Returns:
            :tweets: DataFrame with cleaned keywords.
        """
        tweets['keyword'] = tweets['keyword'].fillna('unknown')
        tweets['keyword'] = tweets['keyword'].apply(lambda x: x.replace('%20', ' '))
        return tweets

    @staticmethod
    def combine_text(row: str) -> str:
        """
        Combine the keyword and text into a single string.
        Arguments:
            :row: DataFrame row containing 'keyword' and 'text' columns.
        Returns:
            :str: Combined string of keyword and text.
        """
        keyword = row['keyword']
        text = row['text']
        return f'keyword: {keyword} text: {text}'

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean the text by removing URLs, mentions, hashtags, and special characters.
        Arguments:
            :text: String containing the text to be cleaned.
        Returns:
            :str: Cleaned text.
        """
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)  # remove full mentions
        text = re.sub(r'#', '', text)  # remove only hashtag sign but leave hashtag text
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text
