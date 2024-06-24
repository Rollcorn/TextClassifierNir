import os
import json


class FeedbackService:
    """
    A class used to handle feedback data.

    Attributes
    ----------
    feedback_file : str
        a string representing the path to the feedback JSON file

    """
    def __init__(self, feedback_file='output/feedback.json'):
        self.feedback_file = feedback_file
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)

    def save_feedback(self, text, model_label, user_label):
        """
        Saves the feedback data to the feedback file. The feedback data consists of the input text, the model's predicted label, and the user's label. If the feedback file does not exist, it is created.

        This method performs the following operations:

        1. Constructs a dictionary with the input text, model's predicted label, and user's label.
        2. Checks if the feedback file exists.
        3. If the feedback file does not exist, it creates the file and writes the feedback data to it.
        4. If the feedback file exists, it reads the existing feedback data, appends the new feedback data to it, and writes the updated feedback data back to the file.

        Args:
            text (str): The input text that was classified.
            model_label (int): The label predicted by the model for the input text.
            user_label (int): The correct label for the input text, as provided by the user.

        Returns:
            None
        """
        feedback_data = {
            "text": text,
            "model_label": model_label,
            "user_label": user_label
        }
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as file:
                json.dump([feedback_data], file)
        else:
            with open(self.feedback_file, 'r+') as file:
                feedbacks = json.load(file)
                feedbacks.append(feedback_data)
                file.seek(0)
                json.dump(feedbacks, file, indent=4)
