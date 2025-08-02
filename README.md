# Feedback Classification using NLP
This project performs intelligent classification of customer feedback using Natural Language Processing (NLP). It classifies over 8,500 manually labeled reviews into multiple categories such as sentiment, recommendation, and query using DistilBERT, a transformer-based model.

## 📂 Project Overview
We developed a system that processes real-world feedback and performs the following tasks:

Sentiment Analysis – Identifies whether the feedback is positive, negative, or neutral.

Recommendation Detection – Flags feedback containing suggestions for improvement.

Query Detection – Detects whether the message is a user query instead of regular feedback.

The model is trained using a manually curated dataset of 8,500+ labeled reviews, ensuring domain relevance and annotation quality.

## 📌 Features
Our model performs multi-label classification on each review to identify the following:

✅ Sentiment Classification
Determines whether the feedback expresses a Positive, Negative, or Neutral sentiment.
Example:
➤ “The service was excellent and the response was quick.” → Sentiment: Positive

💬 Recommendation Detection
Identifies if the feedback includes a suggestion or feature request from the user.
Example:
➤ “You should include a PDF export option.” → Recommendation: Yes

❓ Query Detection
Detects whether the feedback is a question, help request, or not an actual review.
Example:
➤ “How do I change my password?” → Query: Yes

Each review may trigger one or more labels simultaneously, allowing richer and more contextual understanding.


## 🧰 Tech Stack
Language: Python

NLP Model: DistilBERT from HuggingFace Transformers

Libraries:

transformers

pandas, numpy, scikit-learn

streamlit – for the interactive dashboard

oracledb – to connect and fetch reviews from an Oracle SQL database

## 🗃️ Dataset
8,500+ manually labeled customer reviews

Format: CSV (new_corrected_reviews.csv)

## 🤝 Collaborators
Sahil Sohani
Vaibhavi Bhardwaj


