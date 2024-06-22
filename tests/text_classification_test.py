import unittest
from fastapi.testclient import TestClient
from api.text_classifier_controller import app, classifier, logist_classifier, feedback_service


class TestTextClassifierController(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict(self):
        response = self.client.post("/predict", json={"text": "Test text"})
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json())
        self.assertIn('label', response.json())

    def test_logist_predict(self):
        response = self.client.post("/logist/predict", json={"text": "Test text"})
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json())
        self.assertIn('label', response.json())

    def test_feedback(self):
        response = self.client.post("/feedback", json={"text": "Test text", "model_label": 1, "user_label": 0})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "feedback received"})

if __name__ == '__main__':
    unittest.main()