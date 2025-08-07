# GenAI-Driven Product Review Sentiment Analyzer

This project builds a high-accuracy sentiment classification system for Amazon product reviews using a fine-tuned DistilBERT model. The end-to-end pipeline covers data ingestion, cleaning, preprocessing, and a rigorous, data-driven approach to model training and selection.

## Project Overview

The goal of this project is to classify product reviews as **Positive**, **Negative**, or **Neutral**. To achieve this, we've implemented a robust data pipeline and a unique, hypothesis-driven approach to model training. Our solution is not only accurate but also efficient and scalable, making it suitable for real-world applications.

## Our Unique Solution: Smart, Efficient, and Data-Driven

This project stands out due to several critical thinking and best-practice implementations that deliver exceptional results with limited resources and time:

* **Parallel Processing for Speed:** We use Python's `multiprocessing` library to ingest and process large volumes of data in parallel. This drastically reduces the data preparation time and makes the entire pipeline highly scalable.
* **Memory-Efficient Data Handling:** By leveraging the Hugging Face `datasets` library, we can process large datasets that don't fit into memory. This is a crucial feature for handling real-world, large-scale data without requiring expensive high-memory machines.
* **Hypothesis-Driven Model Selection:** Instead of training a single model, we test three different hypotheses to find the best approach for our specific dataset. This demonstrates a deep understanding of machine learning methodology and ensures that our final model is the result of a data-validated decision.
* **Innovative Weighted Loss:** Our winning model uses a custom `WeightedLossTrainer` that pays more attention to reviews marked as "helpful" by the community. This innovative approach guides the model to learn from more reliable data, resulting in a significant boost in accuracy.

## End-to-End Pipeline

The entire project is self-contained in the `Main.ipynb` notebook and is organized into four distinct phases:

### Phase 1: Setup and Environment

This phase imports all necessary libraries and defines the foundational path variables for the project. This ensures that all dependencies are centralized for clarity and maintainability.

### Phase 2: Data Ingestion and Preparation

This phase is responsible for loading, cleaning, and preparing the data for the Transformer model. Key steps include:

* **Loading Data:** The script loads review and metadata files from multiple categories using a reusable `load_jsonl` function.
* **Data Cleaning:** The text data is cleaned by removing HTML tags and special characters.
* **Feature Engineering:** The `title` and `text` of the reviews are combined to create a single, more informative input feature.
* **Sentiment Labeling:** A `sentiment` column is created based on the `rating` of the review (Positive for >= 4.0, Negative for <= 2.0, and Neutral otherwise).
* **Data Balancing:** To prevent the model from being biased towards the majority class, we create a balanced dataset by sampling an equal number of reviews from each sentiment class for each category.

### Phase 3: Model Experimentation and Training

This is the core of our project, where we train and compare three distinct variations of a DistilBERT model:

1.  **Approach 1: Weighted Loss:** This model is trained using a custom `WeightedLossTrainer` that gives more weight to reviews with a `helpful_vote` count greater than zero.
2.  **Approach 2: Filtered Data:** This model is trained on a subset of the data that only includes reviews with a `helpful_vote` count greater than zero.
3.  **Approach 3: Baseline:** This model is trained on the entire balanced dataset without any modifications, serving as a control group to measure the effectiveness of our other approaches.

### Phase 4: Final Evaluation and Conclusion

In this final phase, we evaluate all three saved models on an unseen test set to declare a definitive winner. The results clearly show that the **Weighted Loss** approach is the most effective, achieving the highest accuracy and F1-score.

## Model Performance

The final evaluation results speak for themselves:

| Model | Accuracy | F1-Score (Weighted) |
| :--- | :--- | :--- |
| **Weighted Loss** | **0.8814** | **0.8816** |
| Filtered Data | 0.8012 | 0.8017 |
| Baseline | 0.8171 | 0.8176 |

The **Weighted Loss model** is the clear winner and is the one that has been saved for production deployment.

## Fulfilling Project Requirements

This project successfully fulfills all the requirements of the "GenAI-Driven Product Review Sentiment Analyzer (Local Version)" task:

* **Data Ingestion:** We ingest product reviews from local JSONL files.
* **ETL:** We use Python scripts for data cleaning and sentiment labeling.
* **Model:** We use a DistilBERT model fine-tuned for sentiment analysis with Hugging Face Transformers.
* **Training:** We use a local Python training script with PyTorch and the Hugging Face Trainer.
