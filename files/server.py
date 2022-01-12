import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([6.452020054,6.261979694,1.070622325,1.402182937,
    0.595027924,0.477487415,0.149014473,0.046668742,2.616068125])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'prediction': list(prediction)})


if __name__ == '__main__':
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)