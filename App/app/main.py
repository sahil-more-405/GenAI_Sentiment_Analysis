from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import lime
import lime.lime_text
import numpy as np
from app.predict import predict, tokenizer
from app.gemini_client import get_smart_response # <-- IMPORT THE NEW FUNCTION

app = FastAPI(
    title="GenAI-Driven Product Review Sentiment Analyzer",
    description="A system that classifies product reviews and provides sentiment-aware customer support using a hybrid model approach.",
    version="2.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# In-memory storage for the demo
recent_reviews = []
chat_history = []
negative_streak = 0
conversation_over = False

# This now serves as context for the Gemini model
KNOWLEDGE_BASE = {
    "shipping": "Our standard shipping takes 3-5 business days. Expedited options are available at checkout.",
    "return_policy": "You can return any item within 30 days for a full refund. Please use the portal on our website.",
    "track_order": "To track your order, please use the tracking number sent to your email.",
    "contact_support": "You can reach our support team via the 'Contact Us' page on our website.",
}

# --- All other routes (/ , /predict, /lime-analysis) remain the same ---
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
    global negative_streak, conversation_over

    if text.lower().strip() in ["clear", "reset"]:
        startup_event()
        return templates.TemplateResponse("chatbot.html", {"request": request, "chat_history": chat_history, "conversation_over": conversation_over})

    if conversation_over:
        return templates.TemplateResponse("chatbot.html", {"request": request, "chat_history": chat_history, "conversation_over": conversation_over})

    chat_history.append({"sender": "user", "message": text})

    prediction = predict(text)
    sentiment = prediction['sentiment']

    if sentiment == 'Negative':
        negative_streak += 1
    else:
        negative_streak = 0

    bot_message = ""
    status = ""
    
    # **NEW ADAPTIVE LOGIC WITH GEMINI**
    if negative_streak >= 2:
        bot_message = "I'm very sorry you're having a frustrating experience. To make things right, here is a 15% discount coupon: **WINNER15**. If you'd prefer to speak to a person, you can reach our support helpline at **(555) 123-4567**."
        status = "Escalating to Human Support"
        conversation_over = True
    else:
        # **CALL GEMINI INSTEAD OF THE OLD RAG FUNCTION**
        bot_message = get_smart_response(text, sentiment, KNOWLEDGE_BASE)
        
        if sentiment == 'Negative':
            status = "Sentiment: Negative 😠 (Adapting tone...)"
        elif sentiment == 'Neutral':
            status = "Sentiment: Neutral 😐 (Listening...)"
        else:
            status = "Sentiment: Positive 😊 (Happy to help!)"
    
    chat_history.append({"sender": "bot", "message": bot_message, "status": status})

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
    global negative_streak, conversation_over
    recent_reviews.clear()
    chat_history.clear()
    negative_streak = 0
    conversation_over = False
    chat_history.append({"sender": "bot", "message": "Welcome to our support chat! How can I help you today?", "status": "Sentiment: Neutral 😐 (Listening...)"})