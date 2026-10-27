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

# User inputs for prediction stored in a dictionary
input_data = {}
for col in x.columns:
    # Use the custom label if it exists, otherwise use the original column name
    label = column_labels.get(col, col)
    input_data[col] = st.number_input(f"Enter {label}", min_value=0.0, step=0.1)

# Convert user input to a DataFrame with original column names (so the model doesn't break)
input_df = pd.DataFrame([input_data])

#Predict button 
if st.button("Predict"):
  prediction= model.predict([input_data]) 
  result= ("Diabetic") if prediction[0] == 1 else "Non Diabetic"
  st.success(f"Prediction: {result}")

#model performance metrics
st.subheader("Model Performance")  
y_pred= model.predict(x_test)
accuracy= accuracy_score(y_test,y_pred)
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classifiaction Report:")
st.text(classification_report(y_test,y_pred))





