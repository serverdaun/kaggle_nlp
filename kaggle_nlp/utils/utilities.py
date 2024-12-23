import re

class PreprocessingUtils:

    @staticmethod
    def clean_keywords(tweets):
        tweets['keyword'] = tweets['keyword'].fillna('unknown')
        tweets['keyword'] = tweets['keyword'].apply(lambda x: x.replace('%20', ' '))

        return tweets

    @staticmethod
    def combine_text(row):
        keyword = row['keyword']
        text = row['text']
        return f'keyword: {keyword} text: {text}'

    @staticmethod
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)  # remove full mentions
        text = re.sub(r'#', '', text)  # remove only hashtag sign but leave hashtag text
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text
