import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model1.bin'
dv_file = 'dv.bin'

with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('credit')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        'credit_probability': float(y_pred),
        'credit_probability_round': float(y_pred.round(3))
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)