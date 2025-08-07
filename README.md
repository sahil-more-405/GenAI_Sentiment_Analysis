# GenAI-Driven Product Review Sentiment Analyzer

## Overview

This project showcases a state-of-the-art sentiment analysis system designed to classify Amazon product reviews with exceptional accuracy. It's not just another classifier; it's an end-to-end solution that combines a meticulously fine-tuned model with cutting-edge GenAI features to create a powerful, transparent, and practical application.

Our approach is defined by innovation and technical excellence:

* **Unprecedented Accuracy:** We achieved superior performance by developing a custom **Weighted Loss** function. This unique strategy trains the model to prioritize reviews validated as "helpful" by the community, learning from the most reliable data within a vast and complex dataset.
* **GenAI-Powered Interactions:** The application features a sophisticated chatbot powered by **Google's Gemini**. Using a **Retrieval-Augmented Generation (RAG)** methodology, the chatbot first detects the user's sentiment and then tailors its response to be more empathetic and context-aware, revolutionizing automated customer support.
* **Total Transparency with LIME:** To build trust and demystify our AI, we integrated **LIME (Local Interpretable Model-agnostic Explanations)**. This allows users to see precisely which words in their review influenced the model's prediction, offering complete transparency.
* **Scalable Data Pipeline:** The model was built using a server-friendly pipeline that leverages parallel processing and memory-efficient data handling, capable of processing massive datasets without requiring high-end hardware.

---

## Real-World Applications

The technology demonstrated in this project has immediate, high-impact applications:

* **Automated Customer Support:** Deploy intelligent chatbots that understand user sentiment and respond appropriately.
* **Brand Reputation Management:** Monitor social media and review platforms to instantly gauge public opinion.
* **Product Development:** Gain deep insights from customer feedback to guide product improvements.
* **Market Intelligence:** Analyze competitor reviews to identify market gaps and opportunities.

---

## Project Structure

This repository is organized into two main sub-folders, each documenting a critical phase of the project.

### 1. `Model Pipeline/`

This directory contains the complete data science workflow for creating the sentiment analysis model. Inside, you will find the `Main.ipynb` notebook that covers our entire hypothesis-driven approach, from parallel data ingestion and cleaning to the comparative analysis of three training strategies (Weighted Loss, Filtered Data, and Baseline).

**Explore this folder to understand how our top-performing model was built.**

### 2. `App/`

This directory contains the production-ready, interactive web application. It is a **Dockerized FastAPI application** that serves the fine-tuned model through a REST API. It includes the user interface for sentiment analysis, the Gemini-powered chatbot, and the LIME interpretability dashboard.

**Explore this folder to learn how to deploy and interact with the final application.**

---

## Resources

The project relies on large assets, including the final cleaned dataset, the saved model weights, and a pre-built Docker image for the application. These are essential for reproducing our results and running the application.

* **Download Link:** [Project Assets (Fine-Tuned Model, Dataset, Processed Dataset, Docker Image)](https://drive.google.com/drive/folders/1oFgdMft9bW1NXKLZ3HQbc6PqpgKfDW1j?usp=drive_link)