
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from genres import Genre


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

print("Loading local emotion classifier...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

EMOTIONS = model.config.id2label

# Words and phrases that indicate a movie genre.
# These are genre aliases, not a fixed mood-to-genre mapping.
GENRE_CUES = {
    Genre.ACTION: ["action", "fight", "fighting", "explosive"],
    Genre.ADVENTURE: ["adventure", "adventurous", "explore", "quest"],
    Genre.ANIMATION: ["animation", "animated", "cartoon"],
    Genre.CHILDREN: ["for children", "for kids", "family-friendly"],
    Genre.COMEDY: ["comedy", "funny", "laugh", "cheer me up", "humorous"],
    Genre.CRIME: ["crime", "detective", "criminal", "heist"],
    Genre.DOCUMENTARY: ["documentary", "real-life story", "true story"],
    Genre.DRAMA: ["drama", "emotional story", "deep story"],
    Genre.FANTASY: ["fantasy", "magical", "magic", "mythical"],
    Genre.FILM_NOIR: ["film noir", "noir"],
    Genre.HORROR: ["horror", "scary", "scared", "terrifying", "frightening"],
    Genre.MUSICAL: ["musical", "songs and dancing"],
    Genre.MYSTERY: ["mystery", "puzzle", "whodunit"],
    Genre.ROMANCE: ["romance", "romantic", "love story", "in love"],
    Genre.SCI_FI: ["sci-fi", "science fiction", "space", "futuristic"],
    Genre.THRILLER: ["thriller", "suspense", "edge of my seat"],
    Genre.WAR: ["war movie", "war film", "battlefield"],
    Genre.WESTERN: ["western", "cowboys"],
}

# Used only when the user describes a feeling without specifying
# a movie preference or genre.
EMOTION_FALLBACK = {
    "sadness": [Genre.DRAMA, Genre.ROMANCE],
    "joy": [Genre.COMEDY, Genre.ANIMATION],
    "anger": [Genre.ACTION, Genre.THRILLER],
    "fear": [Genre.THRILLER, Genre.MYSTERY],
    "disgust": [Genre.MYSTERY, Genre.DRAMA],
    "surprise": [Genre.ADVENTURE, Genre.FANTASY],
    "neutral": [Genre.DRAMA, Genre.COMEDY],
}


def detect_emotion(text: str) -> tuple[str, float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    index = int(torch.argmax(probabilities).item())
    emotion = EMOTIONS[index]

    return emotion, float(probabilities[index].item())


def route_to_genres(user_text: str) -> list[Genre]:
    text = user_text.lower().strip()

    if not text:
        return []

    emotion, score = detect_emotion(text)
    print(f"Detected emotion: {emotion} ({score:.2f})")

    # Prefer genres indicated by the user's actual request.
    selected = []

    for genre, cues in GENRE_CUES.items():
        if any(cue in text for cue in cues):
            selected.append(genre)

    # If no genre preference was identified, use the emotion
    # as a basic fallback.
    if not selected:
        selected = EMOTION_FALLBACK.get(
            emotion, [Genre.DRAMA, Genre.COMEDY]
        )

    return selected


if __name__ == "__main__":
    while True:
        text = input(
            "\nDescribe your feelings or the movie you want "
            "(type 'exit' to stop): "
        )

        if text.strip().lower() == "exit":
            break

        genres = route_to_genres(text)

        print("Selected genres:", [genre.value for genre in genres])
