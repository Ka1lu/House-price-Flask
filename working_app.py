from flask import Flask, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessor using relative paths
with open('Exp 5 copy\model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('Exp 5 copy\preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

@app.route('/')
def home():
    # Define a sample house for prediction
    sample_house = pd.DataFrame({
        'area': [3000], 'bedrooms': [4], 'bathrooms': [3], 'stories': [2],
        'mainroad': ['yes'], 'guestroom': ['no'], 'basement': ['yes'],
        'hotwaterheating': ['no'], 'airconditioning': ['yes'], 'parking': [2],
        'prefarea': ['yes'], 'furnishingstatus': ['semi-furnished']
    })

    # Transform the data and predict
    sample_encoded = preprocessor.transform(sample_house)
    predicted_price = model.predict(sample_encoded)
    price = f"{float(predicted_price[0][0]):,.2f}"

    # Render the HTML template
    return render_template('index.html', price=price)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

