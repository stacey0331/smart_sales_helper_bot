from flask import Flask, request, jsonify
import joblib
from script.utils import preprocess_text, sentences_to_embeddings

app = Flask(__name__)

log_reg = joblib.load('./model/svm_model.pkl')
glove_model = joblib.load('./model/glove_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data['sentence']
    preprocessed_sentence = preprocess_text(sentence)
    sentence_embedding = sentences_to_embeddings([preprocessed_sentence], glove_model, vector_size=300)

    prediction = log_reg.predict(sentence_embedding)[0]
    probability = float(log_reg.predict_proba(sentence_embedding)[0][1])

    response = {
        'sentence': sentence,
        'informal_prob': probability
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
