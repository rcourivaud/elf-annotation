from keras_preprocessing.sequence import pad_sequences
import unidecode
from app.config import MAX_LENGTH


def get_prediction_tags(classes, model, features):
    return sorted(zip(classes, model.predict(features)[0]), key=lambda x: x[1], reverse=True)[0:10]


def sequences_from_list_of_text(tokenizer, text_list):
    sequences = tokenizer.texts_to_sequences([unidecode.unidecode(text) for text in text_list])
    return pad_sequences(sequences, maxlen=MAX_LENGTH)