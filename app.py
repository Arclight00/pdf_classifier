import os

import certifi
import nltk
from flask import Flask, request, render_template

from classifier import PdfClassifier

os.environ["SSL_CERT_FILE"] = certifi.where()

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


app = Flask(__name__)

# Initialize your classifier
pdf_classifier = PdfClassifier()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf_url = request.form.get("pdf_url")
        if pdf_url:
            prediction, probability = pdf_classifier.make_predictions(pdf_url)
            return render_template(
                "index.html",
                result=f"Prediction: {prediction}, Probability: {probability}",
            )
        else:
            return render_template("index.html", error="Please enter a URL.")
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
