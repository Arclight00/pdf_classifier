import re
import string

import unidecode
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextProcessing:
    @staticmethod
    def remove_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    @staticmethod
    def remove_emails(text):
        return re.sub(r"([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)", "", text)

    @staticmethod
    def remove_accented_chars(text):
        text = unidecode.unidecode(text)
        return text

    @staticmethod
    def remove_numbers(text):
        return re.sub(r"\d+", "", text)

    @staticmethod
    def remove_slash_with_space(text):
        return text.replace("\\", " ")

    @staticmethod
    def remove_punctuation(text):
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    @staticmethod
    def text_lowercase(text):
        return text.lower()

    @staticmethod
    def remove_whitespace(text):
        return " ".join(text.split())

    @staticmethod
    def remove_stopwords(text):
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return " ".join(filtered_text)

    @staticmethod
    def remove_special_symbols(text):
        pattern = re.compile(r"[\u2122\u00AE\u00A9]", re.UNICODE)
        return pattern.sub("", text)

    @staticmethod
    def lemmatize_words(text):
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(word, pos="v") for word in word_tokens]
        return " ".join(lemmas)

    @staticmethod
    def remove_single_characters(text):
        cleaned_text = re.sub(r"\b\w\b", "", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.strip()

    def preprocess_text(self, text):
        text = self.remove_html_tags(text)
        text = self.remove_emails(text)
        text = self.remove_special_symbols(text)
        text = self.remove_accented_chars(text)
        text = self.remove_numbers(text)
        text = self.remove_slash_with_space(text)
        text = self.remove_punctuation(text)
        text = self.text_lowercase(text)
        text = self.remove_whitespace(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_words(text)
        text = self.remove_single_characters(text)
        return text