import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/full_pipeline_model.pkl')

def user_input_features():
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x==0 else 'Male')
    cp = st.selectbox('Chest Pain Type (cp)', options=[0,1,2,3])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0,1])
    restecg = st.selectbox('Resting ECG results', options=[0,1,2])
    thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', options=[0,1])
    oldpeak = st.number_input('ST depression induced by exercise relative to rest', min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.selectbox('Slope of the peak exercise ST segment', options=[0,1,2])
    ca = st.selectbox('Number of major vessels colored by fluoroscopy', options=[0,1,2,3,4])
    thal = st.selectbox('Thalassemia', options=[3,6,7])  

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

st.title("Heart Disease Risk Prediction")

input_df = user_input_features()

st.subheader('Input Data')
st.write(input_df)

if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:,1]

    risk = 'High Risk' if prediction[0] == 1 else 'Low Risk'
    st.subheader('Prediction')
    st.write(f'Heart Disease Risk: **{risk}**')

    st.subheader('Prediction Probability')
    st.write(f'Probability of Heart Disease: **{prediction_proba[0]:.2f}**')
