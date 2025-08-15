from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoders
with open('customer_churn_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
feature_names = model_data['features_names']

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Define dropdown options for each feature
options = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['No phone service', 'No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'Yes', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['No', 'Yes', 'No internet service'],
    'TechSupport': ['No', 'Yes', 'No internet service'],
    'StreamingTV': ['No', 'Yes', 'No internet service'],
    'StreamingMovies': ['No', 'Yes', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    pred_prob = None
    if request.method == 'POST':
        input_data = {}
        for feature in feature_names:
            if feature in options:
                input_data[feature] = request.form.get(feature)
            else:
                input_data[feature] = float(request.form.get(feature))
        input_df = pd.DataFrame([input_data])
        # Encode categorical features
        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])
        pred = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0]
        prediction = 'Churn' if pred == 1 else 'No Churn'
    return render_template('index.html', options=options, prediction=prediction, pred_prob=pred_prob)

if __name__ == '__main__':
    app.run(debug=True)
