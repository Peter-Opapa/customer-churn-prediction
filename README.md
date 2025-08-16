<img src="images\workflow.png" alt="Workflow Overview" width="800"/>

# Customer Churn Prediction Using Machine Learning

## Project Overview
This project predicts customer churn for a telecom company using machine learning. It demonstrates a complete workflow from data preprocessing and model training to deployment as a web application. The solution enables business stakeholders to input customer data via a user-friendly web interface and receive instant churn predictions.

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Handling class imbalance with SMOTE
- Label encoding for categorical features
- Model training and evaluation (Random Forest, XGBoost, Decision Tree, Logistic Regression, SVM)
- Hyperparameter tuning with GridSearchCV
- Model selection using cross-validation and multiple metrics
- Stratified K-Fold cross-validation for robust evaluation
- Saving trained models and encoders
- Flask web app with dropdowns and numeric inputs for user-friendly predictions
- Production-ready deployment using Gunicorn and Render

## Web App
The app is deployed at: [https://customerchurnprediction-94da.onrender.com](https://customerchurnprediction-94da.onrender.com)

<img src="images\webUI.png" alt="WebUI" width="600"/>

Users can:
- Select categorical features from dropdowns
- Enter numerical features
- Submit the form to get churn prediction and probability

## File Structure
```
├── LICENSE
├── README.md
├── render.yaml
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── deploy-to-render.yml
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── customer_churn_model.pkl
│   └── encoders.pkl
├── notebooks/
│   └── Customer_Churn_Prediction_using_ML.ipynb
├── src/
│   └── app.py
├── templates/
│   └── index.html
├── images/
│   ├── workflow.png
│   ├── webUI.png
│   └── rendersuccess.png
```

## Getting Started
### 1. Clone the repository
```bash
git clone <repo-url>
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app locally
```bash
python app.py
```
Or for production:
```bash
gunicorn app:app
```

### 4. Access the app
Go to `http://localhost:5000` in your browser.

## Model Training Workflow
1. **Data Loading & Inspection:**
	- Load the Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
	- Inspect data structure, missing values, and target distribution.

2. **Exploratory Data Analysis (EDA):**
	- Visualize feature distributions and relationships using pandas, matplotlib, and seaborn.
	- Identify key drivers of churn and data quality issues.

3. **Data Preprocessing:**
	- Handle missing values and convert data types.
	- Encode categorical features using LabelEncoder, saving encoders for deployment.
	- Scale/normalize numerical features as needed.

4. **Class Imbalance Handling:**
	- Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the target classes in training data.

5. **Model Training:**
	- Train multiple models: Decision Tree, Random Forest, XGBoost, Logistic Regression, SVM.
	- Use cross-validation (5-fold) to evaluate each model's accuracy.
	- Select Random Forest as the baseline model based on initial results.

6. **Model Evaluation:**
	- Evaluate the selected model on test data using accuracy, confusion matrix, and classification report.

7. **Hyperparameter Tuning:**
	- Use GridSearchCV to tune Random Forest hyperparameters for improved accuracy.
	- Report best parameters and cross-validation score.

8. **Model Selection & Comparison:**
	- Compare all models using accuracy, F1-score, and ROC-AUC via cross-validation.
	- Summarize results to select the best performing model.

9. **Stratified K-Fold Cross-Validation:**
	- Use StratifiedKFold to ensure robust evaluation and fair representation of classes in each fold.

10. **Saving Artifacts:**
	 - Save the trained model and encoders as pickle files (`customer_churn_model.pkl`, `encoders.pkl`).

11. **Deployment:**
	 - Build a Flask web app for user-friendly predictions.
	 - Deploy the app using Gunicorn and Render, enabling real-time churn prediction via web UI.

**Achievements:**
- Implemented a full ML workflow from EDA to deployment.
- Addressed class imbalance and feature encoding for real-world data.
- Compared multiple models and tuned hyperparameters for optimal performance.
- Built and deployed a production-ready web app for business use.

## Deployment
![Deployment](images\rendersuccess.png)

### Deploying to Render (Main Steps)

1. **Push your code to a Git repository (e.g., GitHub).**
2. **Create a Render account** at [https://render.com](https://render.com).
3. **Create a new Web Service** and connect your repository.
4. **Render will use `requirements.txt` to install dependencies.**
5. **The start command is set in `render.yaml`:**
	```yaml
	start: gunicorn app:app
	```
6. **Click "Create Web Service" to deploy.**
7. **Access your app at the provided Render URL.**

## License
This project is licensed under the MIT License.

## Author
Peter Opapa

---
For questions or contributions, please open an issue or submit a pull request.