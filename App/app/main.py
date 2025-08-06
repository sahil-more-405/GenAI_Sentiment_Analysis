
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.predict import predict

app = FastAPI(
    title="GenAI-Driven Product Review Sentiment Analyzer",
    description="A system that classifies product reviews as Positive, Negative, or Neutral using DistilBERT.",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# In-memory storage for recent reviews
recent_reviews = []

class PredictionRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "reviews": recent_reviews})

@app.post("/predict", response_class=HTMLResponse)
async def predict_text(request: Request, text: str = Form(...)):
    prediction = predict(text)
    review = {"text": text, "sentiment": prediction["sentiment"], "confidence": f"{prediction['confidence']:.2%}"}
    
    # Add to recent reviews and keep only the last 5
    recent_reviews.insert(0, review)
    if len(recent_reviews) > 5:
        recent_reviews.pop()
        
    return templates.TemplateResponse("index.html", {"request": request, "prediction": review, "reviews": recent_reviews})
