from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from service.classificator_model import BertClassificator
from service.feedback_service import FeedbackService

app = FastAPI()
classifier = BertClassificator()
feedback_service = FeedbackService()


class TextRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    text: str
    model_label: int
    user_label: int


predict_router = APIRouter()


@predict_router.post("/predict")
def predict(request: TextRequest):
    try:
        prediction = classifier.predict(request.text)
        label = "Neither" if prediction == 0 else "Yes" if prediction == 1 else "No"
        return {"text": request.text, "prediction": prediction, "label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@predict_router.post("/feedback")
def feedback(request: FeedbackRequest):
    try:
        feedback_service.save_feedback(request.text, request.model_label, request.user_label)
        return {"status": "feedback received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(predict_router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)