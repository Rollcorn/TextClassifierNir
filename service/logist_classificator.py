import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json


class LogistAnswerClassifier:
    """
    This class is used for text classification using a Logistic Regression model.

    Attributes:
        model_path (str): The path to the saved Logistic Regression model.
        vectorizer_path (str): The path to the saved TF-IDF Vectorizer.
        data_path (str): The path to the JSON data file.
        model (LogisticRegression): The Logistic Regression model.
        vectorizer (TfidfVectorizer): The TF-IDF Vectorizer.

    Logistic Regression - a linear model for binary classification that uses the logistic function to model the
    probability of a class.

    TF-IDF Vectorizer - a technique for converting text data into a matrix of TF-IDF features, which are used
    as input to machine learning models.

    TF-IDF features - Term Frequency-Inverse Document Frequency features that represent the importance of a word in
    a document relative to a collection of documents.
    """

    def __init__(self, model_path='model/logistic_model.pkl',
                 vectorizer_path='model/vectorizer.pkl',
                 data_path='resources/data/qas/combined_dataset_with_responses_and_classification.json'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.data_path = data_path
        self.model = None
        self.vectorizer = None

        # Load the model and vectorizer if they exist
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.load_model()
        else:
            self.train_model()
        print("Logist Classifier initialized.")

    def clean_text(self, text):
        """
        Cleans the input text to prepare it for vectorization and classification.

        This method performs the following operations:
        1. Converts the text to lowercase to ensure case-insensitivity.
        2. Removes non-alphanumeric characters to reduce noise.
        3. Strips leading and trailing spaces.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        print("Cleaning text.")
        text = text.lower()
        text = re.sub(r'[^a-zа-я0-9\s]', '', text)
        text = text.strip()
        return text

    def load_data(self):
        """
        Loads the JSON data file and returns a DataFrame with cleaned text and labels.

        This method performs the following operations:\n
        1. Opens the JSON data file specified by the `data_path` attribute.\n
        2. Reads the file line by line, parsing each line as a separate JSON object.\n
        3. Appends each parsed JSON object to a list.\n
        4. Converts the list of JSON objects into a DataFrame.\n
        5. Applies the `clean_text` method to the 'ModelResponse' column of the DataFrame.\n
        6. Converts the 'Classification' column to numerical labels: 'yes' to 1, 'no' to 0, and 'neither' to 2.\n
        7. Returns the DataFrame with 'cleaned_text' and 'label' columns, dropping any rows with missing values.\n

        Returns:
            pd.DataFrame: The DataFrame with cleaned text and labels.

        Raises:
            Exception: If there is an error reading the JSON file.
        """
        print("Loading data.")
        data = []
        try:
            with open(self.data_path, 'r') as file:
                for line in file:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['cleaned_text'] = df['ModelResponse'].apply(self.clean_text)
        df['label'] = df['Classification'].apply(lambda x: 1 if x == 'yes' else 0 if x == 'no' else 2)
        df = df[['cleaned_text', 'label']].dropna()
        return df

    def train_model(self):
        """
        Trains the Logistic Regression model using the loaded data and saves the trained model and vectorizer.

        This method performs the following operations:

        1. Loads the data using the `load_data` method.
        2. Splits the data into training and testing sets.
        3. Initializes a TF-IDF Vectorizer and transforms the training and testing data.
        4. Initializes a Logistic Regression model and fits it on the transformed training data.
        5. Makes predictions on the transformed testing data and prints a classification report.
        6. Saves the trained model and vectorizer using the `save_model` method.

        Raises:
            Exception: If there is an error during training the model or saving the model and vectorizer.
        """
        print("Training model.")
        df = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2,
                                                            random_state=42)

        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model = LogisticRegression(multi_class='ovr', class_weight='balanced')
        self.model.fit(X_train_vec, y_train)

        y_pred = self.model.predict(X_test_vec)
        print(classification_report(y_test, y_pred, target_names=['no', 'yes', 'neither'], zero_division=0))

        self.save_model()

    def save_model(self):
        print("Saving model and vectorizer.")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        with open(self.vectorizer_path, 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)
        print("Model and vectorizer saved.")

    def load_model(self):
        print("Loading model and vectorizer.")
        with open(self.model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        with open(self.vectorizer_path, 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)
        print("Model and vectorizer loaded.")

    def predict(self, text):
        """
            Predicts the class of the input text using the trained Logistic Regression model and returns the predicted class label.

            This method performs the following operations:

            1. Cleans the input text using the `clean_text` method.
            2. Transforms the cleaned text into a vector using the trained TF-IDF Vectorizer.
            3. Makes a prediction using the trained Logistic Regression model on the transformed text.
            4. Returns the predicted class label as an integer (1 for 'yes', 0 for 'no', 2 for 'neither').

            Args:
                text (str): The input text to be classified.

            Returns:
                int: The predicted class label.
        """
        print("Predicting.")
        text_cleaned = self.clean_text(text)
        text_vectorized = self.vectorizer.transform([text_cleaned])
        prediction = self.model.predict(text_vectorized)
        return int(prediction[0])
