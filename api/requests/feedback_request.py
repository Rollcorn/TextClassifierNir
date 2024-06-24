from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    """
    A Pydantic model representing the request body for the /feedback endpoint.

    Attributes
    ----------
    text : str
        The text that was classified.
    model_label : int
        The label predicted by the model for the text.
    user_label : int
        The correct label for the text, as provided by the user.
    """
    text: str
    model_label: int
    user_label: int
