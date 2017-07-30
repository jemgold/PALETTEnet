from keras.models import load_model
from flask import Flask, current_app, request, jsonify
import numpy as np

model = load_model('final.h5')

app = Flask(__name__)
@app.route('/', methods=['POST'])
def predict():
    data = {}
    try:
        data = request.get_json()['data']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400

    return jsonify(data)

    text = np.zeros((1, 2))
    color = np.array([[data['h'], data['s'], data['v']]])

    model.predict()
#     words = []
#     for word in range(2):
#         pred = model.predict([text, color])
#         w = np.argmax(pred[0, word, :])
#         words.append(token_index.get(w))
    
#     fname = ' '.join(words)
app.run(host='0.0.0.0', port=8888, debug=True)
