

# Sentiment Analysis App

A machine learning powered web application that analyzes text sentiment and classifies feedback as **Positive, Negative, or Neutral**.

The application uses a **Logistic Regression model with TF-IDF features** and provides an easy-to-use Streamlit interface for both individual text analysis and bulk PDF feedback analysis.

---

## Features

- Analyze individual text sentiment
- Classify sentiment into:
  - Positive
  - Negative
  - Neutral
- Display model confidence
- Upload PDF files containing multiple feedback entries
- Analyze each feedback separately
- View sentiment distribution
- Filter results by sentiment
- Download analyzed results as CSV
- Clean and responsive Streamlit interface

---

## Demo

### Single Text Analysis

Enter any feedback or review and get its predicted sentiment along with the model confidence.

### PDF Feedback Analysis

Upload a PDF containing feedback entries, preferably one feedback per line.

The application will:

1. Extract text from the PDF
2. Process each feedback entry
3. Predict its sentiment
4. Calculate prediction confidence
5. Display an overall sentiment summary
6. Allow filtering of results
7. Allow downloading the results as CSV

---

## Machine Learning

The sentiment classifier is built using:

- **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Logistic Regression**

### Text preprocessing

Before prediction, the text is cleaned by:

- Converting text to lowercase
- Removing URLs
- Removing mentions
- Removing hashtag symbols
- Removing special characters
- Removing unnecessary whitespace

The same preprocessing pipeline is used during model training and prediction.

---

## Tech Stack

### Frontend / UI
- Streamlit

### Machine Learning
- Python
- Scikit-learn
- Logistic Regression
- TF-IDF

### Data Processing
- Pandas
- Regular Expressions

### PDF Processing
- PyPDF

### Visualization
- Plotly

---

## Live Demo

Try the live application here:

👉 **[Sentiment Analysis App](https://sentiment-analysis-app-iqrr4y3jhtqcjlvignfkf9.streamlit.app/)**

The deployed application allows you to:
- Analyze individual text
- Upload PDF feedback
- Classify feedback as Positive, Negative, or Neutral
- View sentiment distribution
- Download analysis results as CSV


## Project Structure

```text
sentiment-analysis-app/
│
├── app.py
├── train_model.py
├── sentiment_model.pkl
├── Tweets.csv
├── requirements.txt
└── README.md
