from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# Load the trained model from the pickle file
with open('rankpredictions.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route to serve the index.html
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Prediction route
@app.route('/check_rank', methods=['POST'])
def predict():
    try:
        # Get the form data
        name = request.form['name']
        score = float(request.form['score'])  # Ensure score is float if necessary

        # Create polynomial features for the score
        poly_features = PolynomialFeatures(degree=3)
        score_poly = poly_features.fit_transform([[score]])

        # Use the loaded model to predict the rank
        predicted_rank = model.predict(score_poly)[0]

        # Prepare the message based on the prediction
        if predicted_rank > 3500:
            message = "You are not having sufficient skills! Please practice more problems."
        else:
            message = "You have sufficient skills! You can apply for any problem-solving interviews."

        # Return the result as a JSON response
        return jsonify({'status': 'success', 'message': message, 'rank': predicted_rank})

    except Exception as e:
        # Handle any errors and print the exception for debugging
        print(f"Error occurred: {e}")
        return jsonify({'status': 'error', 'message': 'Invalid input or error in prediction: ' + str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(port=3000, debug=True)
