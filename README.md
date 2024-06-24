# Project Title

This project is a text classification system that uses machine learning models to predict the class of a given text. The system is built with Python and uses FastAPI for creating the API endpoints.

## Modules

The project consists of several modules:

1. `text_classifier_controller.py`: This module is the main controller for the API. It defines the endpoints for predicting the class of a text using either the BERT classifier or the Logistic Regression classifier, and for saving feedback data.

2. `text_Request.py` and `feedback_request.py`: These modules define the request bodies for the /predict and /feedback endpoints, respectively.

3. `text_classification_test.py`: This module contains unit tests for the API endpoints.

4. `bert_classificator.py`: This module defines the BERT classifier. It includes methods for loading and saving the model, preprocessing data, training the model, and making predictions.

5. `logist_classificator.py`: This module defines the Logistic Regression classifier. It includes methods for loading and saving the model, cleaning and loading data, training the model, and making predictions.

6. `feedback_service.py`: This module handles feedback data. It includes a method for saving feedback data to a file.

## Technologies

The project uses the following technologies:

- Python: The main programming language used in the project.
- FastAPI: A modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
- Pydantic: A data validation library used for defining the request bodies for the API endpoints.
- Transformers: A state-of-the-art Natural Language Processing library for training and deploying transformers models.
- Scikit-learn: A machine learning library used for training the Logistic Regression classifier.
- Torch: A deep learning framework used in the BERT classifier.
- Unittest: A unit testing framework used for testing the API endpoints.

## Setup

To run this project, you need to have Python installed on your system. After cloning the project, you can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Then, you can start the FastAPI server by running the `text_classifier_controller.py` script:

```bash
python text_classifier_controller.py
```

The server will start on `http://0.0.0.0:8000`. You can interact with the API using a tool like curl or Postman.

## Usage

The API has three endpoints:

- `POST /predict/bert`: Accepts a JSON body with a 'text' field and returns a JSON response with the input text, the predicted class, and the label corresponding to the predicted class using the BERT classifier.

- `POST /predict/logist`: Similar to the /predict endpoint, but uses the Logistic Regression classifier.

- `POST /predict/feedback`: Accepts a JSON body with 'text', 'model_label', and 'user_label' fields. It saves the feedback data and returns a JSON response with a status message.

## Testing

You can run the unit tests by running the `text_classification_test.py` script:

```bash
python text_classification_test.py
```

## Feedback

If you have any feedback or issues, please open an issue in the GitHub repository.