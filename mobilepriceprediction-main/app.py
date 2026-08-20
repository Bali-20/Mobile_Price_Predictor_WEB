from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

# Check if model already trained
MODEL_PATH = 'model.pkl'

if not os.path.exists(MODEL_PATH):
    # Load and preprocess data (only once)
    df = pd.read_csv('mobile_dataset_100k.csv')
    df.drop("Model", axis=1, inplace=True)
    X = df[['Brand', 'RAM', 'Storage', 'Screen Type', 'Condition', 'Model Age']]
    y = df["Price"]

    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    brand_map = {'samsung': 0, 'vivo': 1, 'oppo': 2, 'redmi': 3}
    screen_map = {'soper amoled': 0, 'amoled': 1, 'lcd': 2, 'ips': 3}
    condition_map = {'new': 0, 'good': 1, 'normal': 2, 'bad': 3}
    age_map = {'new': 0, 'old': 1}

    brand = brand_map[request.form['brand'].lower()]
    ram = int(request.form['ram'])
    storage = int(request.form['storage'])
    screen = screen_map[request.form['screen'].lower()]
    condition = condition_map[request.form['condition'].lower()]
    age = age_map[request.form['age'].lower()]

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([[brand, ram, storage, screen, condition, age]])[0]

    return render_template('result.html', price=round(prediction))

if __name__ == '__main__':
    # Railway assigns PORT from environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
