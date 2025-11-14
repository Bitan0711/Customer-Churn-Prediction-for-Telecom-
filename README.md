# Customer Churn Prediction for Telecom Using Machine Learning

A complete end-to-end machine learning project to predict customer churn in a telecom company using the Telco Customer Churn dataset from Kaggle.
This project covers data preprocessing, exploratory analysis, class imbalance handling, ML model training, evaluation, and a final prediction system.

---

## 📌 Project Summary

* Predicts whether a telecom customer is likely to churn (leave the service).
* Uses the Telco Customer Churn dataset with demographic, account, and service-related attributes.
* Handles missing data, encodes categorical variables, and balances classes using SMOTE.
* Trains multiple ML models and selects the best one (Random Forest).
* Provides a user-friendly input-based prediction system.
* Includes model + encoder saving via Pickle for deployment.

---

## 📂 Project Structure


📁 Customer-Churn-Prediction
│
├── 📄 CustomerChurnPrediction.ipynb      # Full code notebook
├── 📄 customer_churn_model.pk1           # Saved Random Forest model
├── 📄 encoders.pk1                       # Saved label encoders
├── 📁 dataset/
│     └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── 📄 README.md                           # Project documentation
└── 📄 requirements.txt                    # Python dependencies


---

## 🧠 Objectives

* Build a machine learning classifier to identify customers likely to churn.
* Preprocess telecom data: handle missing values, encode categories, clean numerical fields.
* Address class imbalance using *SMOTE*.
* Train, compare, and evaluate multiple ML algorithms.
* Save and deploy the trained model for real-time predictions.

---

## 📊 Dataset Overview

* Source: Kaggle – Telco Customer Churn
* Total Records: 7043
* Features:

  * Categorical: gender, partner, dependents, contract type, payment method, etc.
  * Numerical: tenure, monthly charges, total charges
* Issues fixed:

  * Missing values in TotalCharges replaced with 0.0
  * Converted TotalCharges to float
  * Dropped customerID column
  * Identified class imbalance (more “No Churn” cases)

---

## 🔧 Technologies Used

* *Python*
* *Pandas & NumPy* – data cleaning and preprocessing
* *Matplotlib & Seaborn* – data visualization
* *Scikit-learn* – model training & evaluation
* *SMOTE (imblearn)* – oversampling minority class
* *XGBoost*
* *Pickle* – model serialization
* *Google Colab* for development

---

## 🔍 Data Preprocessing

* Checked datatypes, null values, unique values.
* Replaced blanks in TotalCharges and converted to float.
* Encoded categorical features using *LabelEncoder*.
* Stored encoders in a .pk1 file for later use.
* Applied *SMOTE* on training data to fix class imbalance.

---

## 🤖 Model Training

Models trained:

| Model             | CV Accuracy            |
| ----------------- | ---------------------- |
| Decision Tree     | Moderate               |
| XGBoost           | High                   |
| Random Forest     | Highest (selected)     |

* 5-fold cross-validation used for evaluation.
* Random Forest chosen as the best-performing model.
* Model + feature names saved using Pickle.

---

## 🧪 Model Evaluation

After testing on unseen data:

* *Accuracy Score*
* *Confusion Matrix*
* *Precision, Recall, F1-Score* via Classification Report

Results show the model can successfully identify churn with strong accuracy and balanced performance.

---

## 🏗 System Architecture


Data Input → Preprocessing → Encoding → Train-Test Split → SMOTE →
Model Training → Evaluation → Save Model → Load Model → Predict Churn


---

## 🔮 Prediction System

The project includes a simple CLI-based prediction system that:

1. Loads saved model + encoders
2. Accepts user input for all customer features
3. Encodes the inputs
4. Predicts:

   * *Churn* / *No Churn*
   * Prediction probability

---

## 💡 Challenges Faced & Solutions

* *Missing Values*
  → Replaced blanks in TotalCharges

* *Many Categorical Columns*
  → Label encoding + storing encoder objects

* *Class Imbalance*
  → Solved using SMOTE oversampling

* *Ensuring reproducibility*
  → Stored model & encoders using pickle

---

## 🚀 *How to Run This Project*

### 1. Clone the repo


git clone https://github.com/<your-username>/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction


### 2. Install dependencies


pip install -r requirements.txt


### 3. Run the notebook

Open CustomerChurnPrediction.ipynb in Jupyter or Google Colab.

---

## 📜 License

This project is created for academic and learning purposes.

---

## 👤 Author

Bitan Ghosh
B.Tech CSE – Techno India University
