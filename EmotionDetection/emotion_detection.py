
"""Emotion detection using IBM Watson NLP."""

import json
import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion."""

    response = requests.post(
        URL,
        headers=HEADERS,
        json={"raw_document": {"text": text_to_analyze}},
        timeout=30,
    )

    emotions = ("anger", "disgust", "fear", "joy", "sadness")

    if response.status_code == 400:
        return {
            **{emotion: None for emotion in emotions},
            "dominant_emotion": None,
        }

    response.raise_for_status()

    response_dict = json.loads(response.text)
    emotion_scores = response_dict["emotionPredictions"][0]["emotion"]

    scores = {
        emotion: emotion_scores[emotion]
        for emotion in emotions
    }

    scores["dominant_emotion"] = max(scores, key=scores.get)

    return scores
