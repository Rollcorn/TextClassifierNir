import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json
import re


class LogistAnswerClassifier:
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
        print("Cleaning text.")
        text = text.lower()
        text = re.sub(r'[^a-zа-я0-9\s]', '', text)
        text = text.strip()
        return text

    def load_data(self):
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
        print("Training model.")
        df = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

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
        print("Predicting...")
        text_cleaned = self.clean_text(text)
        text_vectorized = self.vectorizer.transform([text_cleaned])
        prediction = self.model.predict(text_vectorized)
        return int(prediction[0])
