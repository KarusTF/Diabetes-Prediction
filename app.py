from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model from the file
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        # Get input values from the form
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = float(request.form['age'])
        pr = float(request.form['pr'])

        input_data = (np.array([pr,glucose, blood_pressure, skin_thickness, insulin, bmi]))
        input_data = np.append(input_data,0)
        input_data = np.append(input_data,age)

        i2d = input_data.reshape(1, -1) 

        prediction = model.predict(i2d)

    if prediction == 1:
            result = 'Diabetes'
    else:
            result = 'No Diabetes'
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)