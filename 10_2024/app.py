from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model and the columns used during training
model = joblib.load('optimized_random_forest_model.pkl')
training_columns = joblib.load('training_columns.pkl')  # Assume you saved the training columns during training

# Initialize the Flask app
app = Flask(__name__)

# Define the home route for rendering the prediction form
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Define the endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        input_data = request.get_json()
        app.logger.debug(f"Received input data: {input_data}")

        # Convert input data to a DataFrame and cast types appropriately
        data = pd.DataFrame([input_data])
        data['year'] = data['year'].astype(int)
        data['engine_power'] = data['engine_power'].astype(float)
        data['ride_km'] = data['ride_km'].astype(int)
        app.logger.debug(f"Converted to DataFrame: {data}")

        # One-hot encode categorical variables
        data = pd.get_dummies(data)

        # Align input data with training columns
        missing_cols = set(training_columns) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        data = data[training_columns]
        app.logger.debug(f"Aligned DataFrame: {data}")

        # Make predictions
        prediction = model.predict(data)
        app.logger.debug(f"Prediction: {prediction}")

        # Return the prediction as a JSON response
        return jsonify({'prediction': round(prediction[0], 2)})
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)})

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)