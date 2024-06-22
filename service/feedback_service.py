import os
import json


class FeedbackService:
    def __init__(self, feedback_file='./../output/feedback.json'):
        self.feedback_file = feedback_file
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)

    def save_feedback(self, text, model_label, user_label):
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
