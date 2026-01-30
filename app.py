# 1 Good (Lower Risk ) 0 Bad (Higher Risk)

# Here we an UI app using streamlit to take inputs from user and predict the credit risk using the trained model

import streamlit as st  # Imports the framework for building the web interface
import pandas as pd  # Imports pandas to structure the user input into a dataframe
import joblib  # Imports the tool to load your saved model and encoders

model = joblib.load("extra_trees_credit_model.pkl")  # Loads your champion Extra Trees model from the saved file
# model = joblib.load("xgb_credit_model.pkl") # Loads your champion XGBoost model from the saved file

encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account']}  # Reloads all saved LabelEncoders into a dictionary using a loop

st.title("Credit Risk Prediction App")  # Adds a large heading to the top of your web page
st.write("Enter applicant information to predict if the credit risk is good or bad")  # Adds a descriptive instruction for the user

# The following lines create interactive input widgets (sliders, dropdowns, and number boxes)
age = st.number_input("Age", min_value = 18, max_value = 80, value = 30)  # Creates a numeric input for age between 18 and 80
sex = st.selectbox("Sex", ["male", "female"])  # Creates a dropdown menu for gender selection
job = st.number_input("Job (0-3)", min_value = 0, max_value = 3, value = 1)  # Creates an input for the job category code
housing = st.selectbox("Housing", ["own", "rent", "free"])  # Creates a dropdown for housing status
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])  # Creates a dropdown for savings levels
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])  # Creates a dropdown for checking account status
credit_amount = st.number_input("Credit Amount", min_value = 0, value = 1000)  # Creates an input for the loan sum
duration = st.number_input("Duration (months)", min_value = 1, value = 12)  # Creates an input for the loan length in months

# Formats the user's inputs into a single-row DataFrame that matches your training data structure
input_df = pd.DataFrame({
    "Age" : [age],  # Uses the numerical age directly
    "Sex" : [encoders["Sex"].transform([sex])[0]],  # Converts the chosen gender text into its trained numerical code
    "Job" : [job],  # Uses the job code directly
    "Housing" : [encoders["Housing"].transform([housing])[0]],  # Encodes the housing selection into a number
    "Saving accounts" : [encoders["Saving accounts"].transform([saving_accounts])[0]],  # Encodes the savings level
    "Checking account" : [encoders["Checking account"].transform([checking_account])[0]],  # Encodes the checking status
    "Credit amount" : [credit_amount],  # Uses the credit sum directly
    "Duration" : [duration]  # Uses the loan duration directly
})

if st.button("Predict Risk"):  # Creates a button and executes the code below only when clicked
    pred = model.predict(input_df)[0]  # Passes the encoded user data to the model to get a 0 or 1 prediction
    
    if pred == 1:  # Checks if the model predicted the "Good" category (assuming 1 is Good)
        st.success("The predicted credit risk is: **GOOD**")  # Displays a green success box with the positive result
    else:  # Handles the "Bad" category prediction
        st.error("The predicted credit risk is: **BAD**")  # Displays a red error box with the negative result