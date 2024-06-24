# Text Classifier

This project is a text classification application built with Python. It uses two different models, a BERT model and a Logistic Regression model, to classify text based on a given dataset.

## Features

- Text classification using BERT and Logistic Regression models.
- Feedback service to save user feedback.
- FastAPI for creating API endpoints.

## Structure

The project is structured as follows:

- `api/text_classifier_controller.py`: This is the main application file. It creates the FastAPI application and defines the API endpoints.
- `service/bert_classificator.py`: This file contains the `BertClassificator` class which is used for text classification using a BERT model.
- `service/logist_classificator.py`: This file contains the `LogistAnswerClassifier` class which is used for text classification using a Logistic Regression model.
- `service/feedback_service.py`: This file contains the `FeedbackService` class which is used to save user feedback.
- `tests/text_classification_test.py`: This file contains the unit tests for the application.

## Setup

To set up the project, you need to have Python installed on your machine. Then, you can install the required dependencies using pip:

```
pip install -r requirements.txt
```

## Usage

To start the application, run the following command:

```
python api/text_classifier_controller.py
```

This will start the FastAPI application on your local machine. You can then access the API endpoints at `http://localhost:8000`.

## Testing

To run the tests, use the following command:

```
python -m unittest tests/text_classification_test.py
```

## Feedback

If you have any feedback or issues, please open an issue in the GitHub repository.
