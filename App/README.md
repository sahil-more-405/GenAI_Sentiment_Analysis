# GenAI-Driven Product Review Sentiment Analyzer (Server Suited)

## Description
This project implements a robust system for classifying product reviews as Positive, Negative, or Neutral using a fine-tuned DistilBERT model. It features an end-to-end pipeline covering model deployment with sentiment scoring APIs and a GenAI-powered chatbot for sentiment-aware customer support. Additionally, it integrates LIME for model interpretability.

**Note:** This repository contains only the application code for deployment and interaction. The data cleaning, preprocessing, and model training phases were conducted in a separate directory.

## Features
*   **Sentiment Analysis:** Classifies product reviews into Positive, Negative, or Neutral using a pre-trained and fine-tuned DistilBERT model.
*   **Interactive Web Interface:** A user-friendly web application built with FastAPI and Jinja2 for submitting reviews and viewing sentiment predictions.
*   **GenAI-Powered Chatbot:** An intelligent chatbot integrated with the Google Gemini API that provides sentiment-aware responses, adapting its tone based on the user's sentiment. It also includes an escalation mechanism for negative user experiences.
*   **Model Interpretability (LIME):** Provides LIME (Local Interpretable Model-agnostic Explanations) analysis for sentiment predictions, helping to understand which words contribute most to a given sentiment.
*   **Dockerized Deployment:** The application is containerized using Docker for easy and consistent deployment.

## Tools and Technologies
*   **Backend Framework:** FastAPI
*   **Frontend:** Jinja2 Templates, HTML, CSS
*   **Sentiment Model:** DistilBERT (from Hugging Face Transformers)
*   **Model Interpretability:** LIME
*   **Generative AI:** Google Gemini API
*   **Containerization:** Docker
*   **Python Libraries:** `torch`, `transformers`, `uvicorn`, `python-dotenv`, `lime`, `numpy`

## Project Structure
```
.
├───.dockerignore
├───Dockerfile
├───requirements.txt
├───app/
│   ├───.env                 # Environment variables (e.g., GEMINI_API_KEY)
│   ├───gemini_client.py     # Handles interaction with the Google Gemini API
│   ├───main.py              # Main FastAPI application, defines routes and logic
│   ├───predict.py           # Contains the sentiment prediction logic using DistilBERT
│   ├───static/
│   │   └───styles.css       # CSS for styling the web interface
│   └───templates/
│       ├───chatbot.html     # HTML template for the chatbot interface
│       ├───index.html       # HTML template for the main sentiment analysis page
│       └───lime-analysis.html # HTML template for LIME explanation display
└───weighted_loss_model/     # Directory containing the fine-tuned DistilBERT model files
    ├───config.json
    ├───model.safetensors
    ├───special_tokens_map.json
    ├───tokenizer_config.json
    ├───tokenizer.json
    ├───training_args.bin
    └───vocab.txt
```

## How it Fulfills Project Requirements

This application fulfills the project requirements by providing a complete, deployable system for product review sentiment analysis with advanced features:

1.  **Sentiment Classification (Positive, Negative, Neutral):** The `app/predict.py` module leverages a fine-tuned DistilBERT model to accurately classify review texts into the specified sentiment categories.
2.  **End-to-End Pipeline (Deployment):** While data ingestion and model training were external, this codebase represents the crucial deployment phase. It packages the trained model within a FastAPI application, making it accessible via a web interface and API endpoints. Dockerization ensures a consistent and reproducible deployment environment.
3.  **Sentiment Scoring APIs:** The `/predict` endpoint in `app/main.py` serves as the core sentiment scoring API, accepting review text and returning the predicted sentiment and confidence.
4.  **GenAI-Driven Functionality:** The integration with the Google Gemini API in `app/gemini_client.py` and its utilization in the `app/main.py` chatbot route (`/chatbot`) demonstrates the "GenAI-Driven" aspect. The chatbot dynamically adjusts its responses based on detected sentiment, providing a more empathetic and intelligent customer support experience.
5.  **Model Interpretability:** The `/lime-analysis` route in `app/main.py` showcases the use of LIME, providing transparency into the model's predictions by highlighting influential words in a review. This addresses the need for understanding *why* a particular sentiment was predicted.

## Setup and Running

### Prerequisites
*   Docker (recommended for easy setup)
*   Python 3.8+
*   `pip` (Python package installer)

### 1. Clone the Repository
```bash
git clone https://github.com/sahil-more-405/GenAI_Sentiment_Analysis.git
cd GenAI_Sentiment_Analysis/Code/App
```

### 2. Set up Gemini API Key (Opional - If Key is not expeired without it also project works fine)
Create a `.env` file inside the `app/` directory (i.e., `app/.env`) and add your Google Gemini API key:
```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```
You can obtain a Gemini API key from the Google AI Studio.

### 3. Running with Docker (Recommended)

#### Option A: Build from Source
This is the easiest way to get the application running if you have Docker installed and want to build the image locally.

```bash
docker build -t sentiment-analyzer .
docker run -p 8000:8000 sentiment-analyzer
```

#### Option B: Load Pre-built Docker Image
If you have a pre-built Docker image as a `.tar` file, you can load it and run the application.

1.  **Download the Docker Image:**
    Download the `gen-ai-sentiment-analyzer.tar` file from [\[YOUR_DOWNLOAD_LINK_HERE\]](https://drive.google.com/drive/folders/1R8bq8KSh7M4tS85x2LhkPUWXEMPQV15k?usp=sharing).

2.  **Load the Image:**
```bash
docker load -i gen-ai-sentiment-analyzer.tar
```

3.  **Run the Container:**
```bash
docker run -p 8000:8000 sentiment-analyzer
```
The application will be accessible at `http://localhost:8000` for both options.

**Note:** To create the `gen-ai-sentiment-analyzer.tar` file from a built image, use the command:
```bash
docker save -o gen-ai-sentiment-analyzer.tar sentiment-analyzer:latest
```

### 4. Running Locally (Python)
If you prefer not to use Docker:

```bash
pip install -r requirements.txt
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The application will be accessible at `http://localhost:8000`.

## Usage

Once the application is running:

*   **Home Page (`/`):** Enter a product review in the text area and click "Analyze Sentiment" to get an instant prediction.
*   **Chatbot (`/chatbot`):** Interact with the AI-powered customer support chatbot. Observe how its responses change based on the sentiment of your messages.
*   **LIME Analysis (`/lime-analysis`):** Input a review to see a LIME explanation, highlighting words that influenced the sentiment prediction.
