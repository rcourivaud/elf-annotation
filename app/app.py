#!flask/bin/python
from flask import Flask, jsonify, request
from keras.models import model_from_json
from keras.backend import clear_session
import json
import pickle
import tensorflow as tf
from app.config import MAX_LABELS
from app.utils import sequences_from_list_of_text

app = Flask(__name__)

clear_session()

json_file = open('./app/res/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./app/res/weights.h5")

with open("./app/res/classes.json", 'r') as f:
    CLASSES = json.load(f)

with open('./app/res/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

graph = tf.get_default_graph()


@app.route('/')
def index():
    return jsonify({"result": "Api is running"})


@app.route('/predict', methods=["POST"])
def predict():
    global graph
    with graph.as_default():
        req = request.get_json()
        question, answer = req["question"], req["answer"]

        question_feature = sequences_from_list_of_text(tokenizer, [question])
        answer_feature = sequences_from_list_of_text(tokenizer, [answer])

        prediction = model.predict([question_feature, answer_feature])[0]
        labels = dict(list(sorted(zip(CLASSES, [float(proba) for proba in prediction]), key=lambda x: x[1], reverse=True))[0:MAX_LABELS])

        return jsonify(labels)
