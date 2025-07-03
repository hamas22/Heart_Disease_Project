# Heart Disease Prediction Project

## Overview
This project aims to build a machine learning model to predict the risk of heart disease based on patient clinical data. It includes data preprocessing, feature engineering, model training, evaluation, and a Streamlit web app for user interaction.

## Dataset
- Dataset used: Cleveland Heart Disease Dataset (processed.cleveland.data)
- Number of features used: 13 clinical features plus target variable
- Source: UCI Machine Learning Repository

## Project Structure

 File Structure 
Heart_Disease_Project/ 
│── data/ 
│   
├── heart_disease.csv 
│── notebooks/ 
│   
│   
│   
│   
│   
│   
├── 01_data_preprocessing.ipynb 
├── 02_pca_analysis.ipynb 
├── 03_feature_selection.ipynb 
├── 04_supervised_learning.ipynb 
├── 05_unsupervised_learning.ipynb 
├── 06_hyperparameter_tuning.ipynb 
│── models/ 
│   
├── final_model.pkl 
│── ui/ 
│   
├── app.py (Streamlit UI) 
│── deployment/ 
│   
├── ngrok_setup.txt 
│── results/ 
│   
├── evaluation_metrics.txt 
│── README.md 
│── requirements.txt 
│── .gitignore


## How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt


Run the Streamlit app
```bash
python -m streamlit run ui/app.py
```

eatures Used
Age

Sex

Chest pain type (cp)

Resting blood pressure (trestbps)

Serum cholesterol (chol)

Fasting blood sugar (fbs)

Resting ECG results (restecg)

Maximum heart rate achieved (thalach)

Exercise induced angina (exang)

ST depression induced by exercise (oldpeak)

Slope of the peak exercise ST segment (slope)

Number of major vessels colored by fluoroscopy (ca)

Model
Model pipeline includes data scaling, PCA, and classification (e.g., Random Forest).

Achieved accuracy: around 83%

Author
Hamas Mohamed

