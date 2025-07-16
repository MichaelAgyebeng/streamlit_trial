import streamlit as st
import pandas as pd
import joblib
import pickle

# with open("C:/Users/T PLUG/Downloads/data 1 (4)/logistic_regression_model.pkl", "rb") as file:
#     model = pickle.load(file)
# Load the model
model = joblib.load("C:/Users/T PLUG/Downloads/data 1 (4)/logistic_regression_model.pkl")

# UI
st.title("Term Deposit Subscription Predictor")

# User input
st.header("Client Info")
age = st.number_input("Age", 18, 100, 35)
job = st.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired',
 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
education = st.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y',
       'professional.course', 'unknown', 'university.degree', 'illiterate'])
default = st.selectbox("Has Default?", ['yes', 'no', 'unknown'])
housing = st.selectbox("Has Housing Loan?", ['yes', 'no', 'unknown'])
loan = st.selectbox("Has Personal Loan?", ['yes', 'no', 'unknown'])
duration = st.number_input("Call Duration (seconds)", 0, 10000, 100)
campaign = st.number_input("Campaign Contacts", 1, 50, 1)
pdays = st.number_input("Days Since Last Contact", -1, 999, -1)
previous = st.number_input("Previous Contacts", 0, 100, 0)
poutcome = st.selectbox("Previous Outcome", ['failure', 'success', 'nonexistent', 'unknown'])


# Get the expected feature names from the model
if hasattr(model, "feature_names_in_"):
    expected_features = model.feature_names_in_
else:
    raise ValueError("The model does not have 'feature_names_in_' attribute. Cannot build proper input.")


if st.button("Predict"):
    input_df = pd.DataFrame([{
        'age': age,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'job_' + job: 1,
        'marital_' + marital: 1,
        'education_' + education: 1,
        'default_yes': 1 if default == 'yes' else 0,
        'housing_yes': 1 if housing == 'yes' else 0,
        'loan_yes': 1 if loan == 'yes' else 0,
        'poutcome_' + poutcome: 1,
    }])

    # # Fill missing dummies
    # all_features = model.feature_names_in_
    # for col in all_features:
    #     if col not in input_df.columns:
    #         input_df[col] = 0
    # input_df = input_df[all_features]

    # pred = model.predict(input_df)[0]
    # st.success(f"Prediction: {'Subscribed' if pred == 1 else 'Not Subscribed'}")

    # Fill missing features with 0 to match model's expected input
    full_input = {feature: input_df.get(feature, 0) for feature in expected_features}

    # Create a DataFrame with the correct column order
    input_df = pd.DataFrame([full_input])

    # Make prediction
    pred = model.predict(input_df)

    # Output the result
    st.success(f"Prediction: {'Subscribed' if pred == 1 else 'Not Subscribed'}")
