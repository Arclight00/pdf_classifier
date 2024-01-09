from io import BytesIO

import pdfplumber
import requests
import spacy
from joblib import load

from preprocessing import TextProcessing

model_filename = "random_forest.joblib"


class PdfClassifier:
    def __init__(self):
        self.model = load(model_filename)

    def get_word2vec(self, text):
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)
        return doc.vector

    def extract_text_from_pdf(self, url):
        try:
            # Fetch the PDF from the URL
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Error fetching the PDF: {e}"

        try:
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
        except Exception as e:
            return f"Error processing the PDF: {e}"

        extracted_text = "\n".join(filter(None, pages))

        if not extracted_text:
            return "No text found in the PDF."

        return extracted_text

    def make_predictions(self, pdf_url):
        text_process = TextProcessing()
        text = self.extract_text_from_pdf(pdf_url)
        text = text_process.preprocess_text(text)
        vec_text = self.get_word2vec(text)
        vec_text = vec_text.reshape(1, -1)
        prediction = self.model.predict(vec_text)
        probability_estimates = self.model.predict_proba(vec_text)
        prediction_label = "lighting" if prediction == 1 else "non-lighting"
        print(f"Prediction: {prediction_label}")
        print("Probability Estimates:", probability_estimates)
        return prediction_label, probability_estimates