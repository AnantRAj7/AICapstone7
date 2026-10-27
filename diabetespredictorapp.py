import streamlit as st
import pandas as pd
import sys
import importlib.util
import streamlit as st
import sklearn

st.write("Python version:", sys.version)
st.write("Python executable:", sys.executable)
st.write("sklearn found:", importlib.util.find_spec("sklearn"))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#load dataset 
data = pd.read_csv('diabetes.csv')

#split data into features (X) and target (Y)
x=data.drop('Outcome', axis=1)
y=data['Outcome']

#split the dataset into training and testing data sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

#streamlit app interface 
st.title("Diabetes Prediction App")

# Medical Disclaimer Alert Box
st.warning("**Disclaimer:** This tool is intended for educational and awareness purposes only. It does not provide medical diagnosis or advice. Please consult a healthcare professional for accurate medical guidance.")
st.write("Enter the following details to predict the likelihood of diabetes")

# Define custom readable names for your features
column_labels = {
    'Pregnancies': 'Number of Pregnancies',
    'Glucose': 'Blood Glucose Level',
    'BloodPressure': 'Blood Pressure (mm Hg)',
    'SkinThickness': 'Skin Thickness (mm)',
    'Insulin': 'Insulin Level (mu U/ml)',
    'BMI': 'Body Mass Index (BMI)',
    'DiabetesPedigreeFunction': 'Diabetes Genetics',
    'Age': 'Age (years)'
}

# Define custom readable names and realistic ranges (min, max, default value, step)
feature_configs = {
    'Pregnancies': {'label': 'Number of Pregnancies', 'min': 0.0, 'max': 20.0, 'default': 1.0, 'step': 1.0},
    'Glucose': {'label': 'Blood Glucose Level', 'min': 0.0, 'max': 300.0, 'default': 120.0, 'step': 1.0},
    'BloodPressure': {'label': 'Blood Pressure (mm Hg)', 'min': 0.0, 'max': 200.0, 'default': 70.0, 'step': 1.0},
    'SkinThickness': {'label': 'Skin Thickness (mm)', 'min': 0.0, 'max': 100.0, 'default': 20.0, 'step': 1.0},
    'Insulin': {'label': 'Insulin Level', 'min': 0.0, 'max': 900.0, 'default': 80.0, 'step': 1.0},
    'BMI': {'label': 'Body Index (BMI)', 'min': 0.0, 'max': 70.0, 'default': 25.0, 'step': 0.1},
    'DiabetesPedigreeFunction': {'label': 'Diabetes Pedigree Function', 'min': 0.0, 'max': 3.0, 'default': 0.5, 'step': 0.01},
    'Age': {'label': 'Age (years)', 'min': 1.0, 'max': 120.0, 'default': 30.0, 'step': 1.0}
}

# User inputs for prediction stored in a dictionary
input_data = {}
for col in x.columns:
    config = feature_configs.get(col, {'label': col, 'min': 0.0, 'max': 1000.0, 'default': 0.0, 'step': 0.1})
    
    input_data[col] = st.number_input(
        f"Enter {config['label']}", 
        min_value=config['min'], 
        max_value=config['max'], 
        value=config['default'], 
        step=config['step']
    )

# Convert user input to a DataFrame
input_df = pd.DataFrame([input_data])

# Convert user input to a DataFrame with original column names (so the model doesn't break)
input_df = pd.DataFrame([input_data])

# Predict button 
if st.button("Predict"):
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.markdown("<h3 style='color: red;'>Prediction: Diabetic</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>Prediction: Non-Diabetic</h3>", unsafe_allow_html=True)

#model performance metrics
st.subheader("Model Performance")  
y_pred= model.predict(x_test)
accuracy= accuracy_score(y_test,y_pred)
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classifiaction Report:")
st.text(classification_report(y_test,y_pred))





