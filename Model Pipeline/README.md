# GenAI-Driven Product Review Sentiment Analyzer (Model Pipeline)

## Overview

This document outlines the end-to-end pipeline for processing the Amazon Product Reviews dataset and training a sentiment analysis model. This pipeline is a crucial component of the GenAI-Driven Product Review Sentiment Analyzer project, providing the core intelligence for the sentiment scoring API.

## Dataset

*   **Dataset Name:** Amazon Product Reviews
*   **Source:** Amazon Reviews Dataset

## End-to-End Pipeline

The model pipeline is designed to be efficient and effective, transforming raw product reviews into a powerful sentiment analysis model.

### 1. Data Ingestion & Pre-processing

The initial phase focuses on ingesting the raw data and preparing it for model training. This involves:

*   **Data Loading:** Importing the dataset from local CSV files.
*   **Data Cleaning:**
    *   Handling missing values to ensure data integrity.
    *   Removing duplicate entries to prevent bias.
    *   Standardizing text by converting to lowercase, removing special characters, and eliminating stopwords.
*   **Feature Engineering:** Creating a "sentiment" column based on the "Score" column, mapping ratings to "Positive," "Negative," and "Neutral" categories.

### 2. Model Selection

For this task, we have chosen **DistilBERT**, a distilled version of BERT. This choice was driven by the following considerations:

*   **Efficiency:** DistilBERT is smaller and faster than larger models like BERT-base or RoBERTa, leading to significantly reduced training time and computational resource requirements.
*   **Performance:** Despite its smaller size, DistilBERT retains most of the performance of the original BERT model, making it an ideal choice for this application.

### 3. Training

The model is fine-tuned using the Hugging Face Transformers library with PyTorch as the backend. The training process involves:

*   **Tokenization:** Converting the cleaned text into a format that the model can understand.
*   **Training Loop:** Utilizing the Hugging Face Trainer to fine-tune the DistilBERT model on our pre-processed dataset.

### 4. Evaluation

The model's performance is evaluated using a classification report and a confusion matrix. The results demonstrate a high level of accuracy in classifying reviews as Positive, Negative, or Neutral.

## Unique Aspects & Critical Thinking

Our solution is unique in its focus on achieving the best possible results with limited resources and time.

*   **Resource Efficiency:** The use of DistilBERT allows us to train a high-performing model without the need for expensive, high-end hardware. This makes our solution more accessible and easier to deploy.
*   **Comprehensive Pre-processing:** Our thorough data cleaning and pre-processing pipeline ensures that the model is trained on high-quality data, leading to more robust and reliable predictions.
*   **End-to-End Automation:** The entire pipeline, from data ingestion to model training, is automated in the `Main.ipynb` notebook, making it easy to reproduce and retrain the model with new data.

## Fulfilling Project Requirements

This model pipeline directly fulfills the core requirements of the project by:

*   **Providing a trained sentiment analysis model:** The output of this pipeline is a fine-tuned DistilBERT model capable of accurately classifying product reviews.
*   **Enabling the sentiment scoring API:** The trained model is the key component of the FastAPI-based sentiment scoring API, allowing the application to provide real-time sentiment analysis.
*   **Demonstrating an end-to-end solution:** This pipeline, in conjunction with the application code in the `App` folder, provides a complete, end-to-end solution for sentiment analysis, from data processing to deployment.
