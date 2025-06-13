import torch
from flask import Flask, request, jsonify, render_template_string
import logging
from model_class1 import SentenceClassifier
import json
from flask_cors import CORS

MODEL_NAME = "intfloat/multilingual-e5-small"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)
model = SentenceClassifier()
model.load_from_file('/Studying/AAA/Term2/v5_small_last_ep.pt', MODEL_NAME, device=device)

def convert_col(data):
    for i in range(len(data)):
        if data[i] == 0:
            data[i] = "Нейтральное"
        elif data[i] == 1:
            data[i] = "Оскорбление"
        elif data[i] == 2:
            data[i] = "Угроза"
        elif data[i] == 3:
            data[i] = "Домогательство"
    return data

@app.route('/read_msgs', methods=['POST'])
def read_msg_many():
    data = request.get_json()
    if not data:
        return 'No data received', 400
    
    data1 = data["messages"]
    prediction = model.predict_tone(data1)
    prediction = convert_col(prediction)
    for i in range(len(data1)):
        if len(data1[i]) > 55:
            data1[i] = data1[i][: 55] + "..."
    prediction = {"messages": data1, "tone": prediction}
    return json.dumps(prediction)

if __name__ == '__main__':
    logging.basicConfig(
        format='[%(levelname)s] [%(asctime)s] %(message)s',
        level=logging.INFO,
    )

    app.run(host='0.0.0.0', port=8080, debug=True)
