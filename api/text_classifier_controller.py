from fastapi import FastAPI, HTTPException, APIRouter

from api.requests.feedback_request import FeedbackRequest
from api.requests.text_Request import TextRequest
from service.bert_classificator import BertClassificator
from service.feedback_service import FeedbackService
from service.logist_classificator import LogistAnswerClassifier


app = FastAPI()
classifier = BertClassificator()
logist_classifier = LogistAnswerClassifier()

feedback_service = FeedbackService()
predict_router = APIRouter()


@predict_router.post("/bert")
def predict(request: TextRequest):
    """
    Endpoint for predicting the class of a text using the BERT classifier.

    This endpoint accepts a POST request with a JSON body containing a 'text' field. It returns a JSON response
    containing the input text, the predicted class, and the label corresponding to the predicted class.

    Args:
        request (TextRequest): The request body containing the text to be classified.

    Returns:
        dict: A dictionary containing the input text, the predicted class, and the label corresponding to the predicted
        class.
    """
    print("Request: ", request.text)
    try:
        prediction = classifier.predict(request.text)
        label = "Neither" if prediction == 0 else "Yes" if prediction == 1 else "No"
        return {"text": request.text, "prediction": prediction, "label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@predict_router.post("/logist")
def predict(request: TextRequest):
    """
    Endpoint for predicting the class of a text using the logistic regression classifier.

    This endpoint accepts a POST request with a JSON body containing a 'text' field. It returns a JSON response
    containing the input text, the predicted class, and the label corresponding to the predicted class.

    Args:
        request (TextRequest): The request body containing the text to be classified.

    Returns:
        dict: A dictionary containing the input text, the predicted class, and the label corresponding to the predicted
        class.
    """
    print("Request: ", request.text)
    try:
        prediction = logist_classifier.predict(request.text)
        label = "Neither" if prediction == 0 else "Yes" if prediction == 1 else "No"
        return {"text": request.text, "prediction": prediction, "label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@predict_router.post("/feedback")
def feedback(request: FeedbackRequest):
    """
    Endpoint for saving feedback data.

    This endpoint accepts a POST request with a JSON body containing a 'text' field, a 'model_label' field, and a
    'user_label' field. It returns a JSON response containing a status message.

    Args:
        request (FeedbackRequest): The request body containing the feedback data.

    Returns:
        dict: A dictionary containing a status message.
    """
    print("Request: ", request.text)
    try:
        feedback_service.save_feedback(request.text, request.model_label, request.user_label)
        return {"status": "feedback received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(predict_router, prefix="/predict")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
