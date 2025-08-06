from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import lime
import lime.lime_text
import numpy as np
from app.predict import predict, tokenizer

app = FastAPI(
    title="GenAI-Driven Product Review Sentiment Analyzer",
    description="A system that classifies product reviews as Positive, Negative, or Neutral using DistilBERT.",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# In-memory storage
recent_reviews = []
chat_history = []
cumulative_sentiment = 0
conversation_over = False

# Simple Knowledge Base for RAG
KNOWLEDGE_BASE = {
    "shipping": "Our standard shipping takes 3-5 business days. Expedited shipping is available for an additional cost.",
    "return policy": "You can return any item within 30 days for a full refund. Please visit our returns page for more details.",
    "track order": "To track your order, please use the tracking number sent to your email or visit the 'My Orders' section of your account.",
    "contact": "You can contact our support team via the 'Contact Us' page on our website.",
    "default": "I'm not sure how to answer that. Could you please rephrase your question?"
}

def retrieve_from_kb(query: str) -> str:
    """A simple retrieval function for our RAG-like structure."""
    query = query.lower()
    for keyword, answer in KNOWLEDGE_BASE.items():
        if keyword in query:
            return answer
    return KNOWLEDGE_BASE["default"]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "reviews": recent_reviews})

@app.post("/predict", response_class=HTMLResponse)
async def predict_text(request: Request, text: str = Form(...)):
    prediction = predict(text)
    review = {"text": text, "sentiment": prediction["sentiment"], "confidence": f"{prediction['confidence']:.2%}"}

    recent_reviews.insert(0, review)
    if len(recent_reviews) > 5:
        recent_reviews.pop()

    return templates.TemplateResponse("index.html", {"request": request, "prediction": review, "reviews": recent_reviews})

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request, "chat_history": chat_history, "conversation_over": conversation_over})

@app.post("/chatbot", response_class=HTMLResponse)
async def chat(request: Request, text: str = Form(...)):
    global cumulative_sentiment, conversation_over

    if text.lower() in ["clear", "reset"]:
        startup_event()
        return templates.TemplateResponse("chatbot.html", {"request": request, "chat_history": chat_history, "conversation_over": conversation_over})

    if conversation_over:
        return templates.TemplateResponse("chatbot.html", {"request": request, "chat_history": chat_history, "conversation_over": conversation_over})

    chat_history.append({"sender": "user", "message": text})

    # 1. Analyze sentiment of the user's message
    prediction = predict(text)
    sentiment = prediction['sentiment']

    # 2. Update cumulative sentiment score
    if sentiment == 'Positive':
        cumulative_sentiment += 1
    elif sentiment == 'Negative':
        cumulative_sentiment -= 1

    # 3. Determine bot's response based on sentiment
    if cumulative_sentiment < -1:
        bot_message = "I'm sorry that I wasn't able to help you. I can connect you to a human agent to resolve this issue."
        conversation_over = True
    else:
        # 4. Retrieve answer from Knowledge Base (RAG)
        bot_message = retrieve_from_kb(text)

    # 5. Determine conversation status
    if cumulative_sentiment <= -1:
        status = "Negative"
    elif cumulative_sentiment >= 1:
        status = "Positive"
    else:
        status = "Neutral"

    chat_history.append({"sender": "bot", "message": bot_message, "status": f"Conversation Status: {status}"})

    return templates.TemplateResponse("chatbot.html", {"request": request, "chat_history": chat_history, "conversation_over": conversation_over})


@app.get("/lime-analysis", response_class=HTMLResponse)
async def lime_page(request: Request):
    return templates.TemplateResponse("lime-analysis.html", {"request": request})

@app.post("/lime-analysis", response_class=HTMLResponse)
async def get_lime_analysis(request: Request, text: str = Form(...)):
    explainer = lime.lime_text.LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])

    def predictor(texts):
        outputs = [predict(t) for t in texts]
        raw_confidences = [o['raw_confidence'] for o in outputs]
        return np.array(raw_confidences)

    explanation = explainer.explain_instance(text, predictor, num_features=10, num_samples=500)

    return templates.TemplateResponse("lime-analysis.html", {"request": request, "explanation": explanation.as_html()})

@app.on_event("startup")
async def startup_event():
    global cumulative_sentiment, conversation_over
    recent_reviews.clear()
    chat_history.clear()
    cumulative_sentiment = 0
    conversation_over = False