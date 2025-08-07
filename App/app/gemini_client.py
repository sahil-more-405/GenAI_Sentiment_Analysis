import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=api_key)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

def get_smart_response(user_query: str, sentiment: str, knowledge_base: dict) -> str:
    """
    Generates a smart, sentiment-aware response using the Gemini API.
    """
    # Convert knowledge base to a string for the prompt
    knowledge_context = "\n".join([f"- {key.replace('_', ' ')}: {value}" for key, value in knowledge_base.items()])

    # This is the core of the "GenAI" feature: a well-crafted prompt
    prompt = f"""
    You are "Sam", an advanced, empathetic customer support AI. Your goal is to provide helpful answers while managing the user's emotional state.

    **Analysis of User's Message:**
    - User's Message: "{user_query}"
    - Detected Sentiment: **{sentiment}**

    **Your Knowledge Base:**
    {knowledge_context}

    **Your Task:**
    Based on the user's message and their sentiment, generate a concise, helpful, and empathetic response.
    - If the sentiment is **Negative**, your tone must be extra apologetic and reassuring before answering.
    - If the sentiment is **Positive**, your tone should be cheerful and encouraging.
    - If the sentiment is **Neutral**, be straightforward and helpful.
    - Use the knowledge base to answer the question. If the user's question is not covered, politely state you can help with the topics you know about.
    - **Do not** mention that you are an AI or talk about sentiment analysis. Just act the part.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "I'm having a little trouble connecting right now. Please try again in a moment."