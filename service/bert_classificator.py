import os
import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import zipfile


def load_data():
    data = []
    try:
        with open('resources/data/qas/combined_dataset_with_responses_and_classification.json', 'r') as file:
            for line in file:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
            print("Data loaded")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return []

    return pd.DataFrame(data)


class BertClassificator:
    """
        A class used to represent a BERT Classifier for text classification tasks.

        Attributes
        ----------
        data_path : str
            a string representing the path to the JSON data file
        model_name : str
            a string representing the name of the pre-trained BERT model to use
        num_labels : int
            an integer representing the number of labels in the classification task
        model_dir : str
            a string representing the directory where the trained model and tokenizer will be saved
        device : torch.device
            a torch.device object representing the device (CPU or GPU) where the computations will be performed
        tokenizer : transformers.BertTokenizer
            a BertTokenizer object from the transformers library
        model : transformers.BertForSequenceClassification
            a BertForSequenceClassification object from the transformers library
    """
    def __init__(self,
                 data_path='./../resources/qas/combined_dataset_with_responses_and_classification.json',
                 model_name='bert-base-multilingual-cased',
                 num_labels=3,
                 model_dir='model'
                 ):
        self.data_path = data_path
        self.model_name = model_name
        self.num_labels = num_labels
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)

        # Load model if available, else train
        if not self.load_model():
            self.data = load_data()
            self.train_dataloader, self.val_dataloader = self.prepare_datasets()
            self.train()
        print("BertClassificator initialized.")

    def load_model(self):
        """
        Loads the trained BERT model and tokenizer from the specified directory if they exist.

        This method performs the following operations:
        1. Constructs the paths to the model and tokenizer using the `model_dir` attribute.
        2. Checks if the model and tokenizer files exist at the constructed paths.
        3. If they exist, loads the model parameters from the model file into the `model` attribute and loads the tokenizer from the tokenizer file into the `tokenizer` attribute.
        4. Returns True if the model and tokenizer were successfully loaded, False otherwise.

        Returns:
            bool: True if the model and tokenizer were successfully loaded, False otherwise.
        """
        model_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer')
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            print("Model and tokenizer loaded.")
            return True
        print("Model and tokenizer not found.")
        return False

    def save_model(self):
        """
        Saves the trained BERT model and tokenizer to the specified directory.

        This method performs the following operations:
        1. Checks if the directory specified by the `model_dir` attribute exists, and creates it if it does not.
        2. Constructs the paths to the model and tokenizer files using the `model_dir` attribute.
        3. Saves the `model` attribute's parameters to the model file and saves the `tokenizer` attribute to the tokenizer file.
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer')
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        print("Model and tokenizer saved.")

    def preprocess_data(self, text_list, max_length=128):
        """
        Preprocesses the input text list for BERT input.

        This method performs the following operations:

        1. Initializes empty lists for input IDs and attention masks.
        2. Iterates over the input text list.
        3. For each text in the list, encodes the text into input IDs and attention masks using the `tokenizer` attribute, and appends them to the respective lists.
        4. Converts the lists of input IDs and attention masks into tensors and returns them.

        Args:
            text_list (list): The list of texts to be preprocessed.
            max_length (int, optional): The maximum length for the encoded texts. Defaults to 128.

        Returns:
            tuple: A tuple containing the tensors of input IDs and attention masks.
        """
        input_ids, attention_masks = [], []

        for text in text_list:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        print("Data preprocessed.")
        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

    def prepare_datasets(self):
        """
            Prepares the training and validation datasets.

            This method performs the following operations:
            1. Preprocesses the 'ModelResponse' column of the `data` attribute into input IDs and attention masks using the `preprocess_data` method.
            2. Converts the 'Classification' column of the `data` attribute into numerical labels and stores them in a tensor.
            3. Combines the tensors of input IDs, attention masks, and labels into a dataset.
            4. Splits the dataset into training and validation datasets.
            5. Wraps the training and validation datasets in DataLoaders and returns them.

            Returns:
                tuple: A tuple containing the DataLoaders for the training and validation datasets.
        """
        input_ids, attention_masks = self.preprocess_data(self.data['ModelResponse'].to_list())
        labels = self.data['Classification'].apply(lambda x: 0 if x == 'neither' else 1 if x == 'yes' else 2).values
        labels = torch.tensor(labels).to(self.device)

        train_size = int(0.8 * len(self.data))
        val_size = len(self.data) - train_size

        train_dataset, val_dataset = random_split(TensorDataset(input_ids, attention_masks, labels), [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        print("Datasets prepared.")
        return train_dataloader, val_dataloader

    def train(self, epochs=4, learning_rate=2e-5):
        """
        Trains the BERT model on the training dataset.

        This method performs the following operations:

        1. Initializes the AdamW optimizer with the model parameters and the specified learning rate.
        2. Sets the model to training mode.
        3. Iterates over the training data for the specified number of epochs.
        4. For each batch in the training data, it performs a forward pass, computes the loss, performs a backward pass, and updates the model parameters.
        5. After each epoch, it computes the average training loss and prints it.
        6. After training, it evaluates the model on the validation dataset and saves the model and tokenizer.

        Args:
            epochs (int, optional): The number of times to iterate over the training data. Defaults to 4.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 2e-5.
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        print("Start training")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in self.train_dataloader:
                b_input_ids, b_input_mask, b_labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                self.model.zero_grad()
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.2f}")

            self.evaluate()

        self.save_model()

    def evaluate(self):
        """
        Evaluates the BERT model on the validation dataset.

        This method performs the following operations:

        1. Sets the model to evaluation mode.
        2. Iterates over the validation data.
        3. For each batch in the validation data, it performs a forward pass, computes the loss, and computes the accuracy.
        4. After evaluation, it computes the average validation loss and accuracy and prints them.
        """
        self.model.eval()
        val_accuracy, val_loss = 0, 0

        for batch in self.val_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy += accuracy

        avg_val_accuracy = val_accuracy / len(self.val_dataloader)
        avg_val_loss = val_loss / len(self.val_dataloader)
        print(f'Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {avg_val_accuracy:.2f}%')

    def predict(self, text):
        """
        Predicts the class of the input text using the trained BERT model.

        This method performs the following operations:

        1. Sets the model to evaluation mode.
        2. Preprocesses the input text for BERT input.
        3. Performs a forward pass with the preprocessed input.
        4. Computes the logits from the model output.
        5. Returns the class with the highest logit as the predicted class.

        Args:
            text (str): The input text to be classified.

        Returns:
            int: The predicted class label.
        """
        self.model.eval()
        inputs = self.preprocess_data([text])
        input_ids, attention_masks = inputs[0].to(self.device), inputs[1].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        print("Prediction made: ", prediction)
        return prediction
