import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num' 
]

df = pd.read_csv('heart+disease/processed.cleveland.data', header=None, names=column_names, na_values='?')

df = df.dropna()

X = df.drop('num', axis=1)
y = df['num'].apply(lambda x: 1 if x > 0 else 0)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'models/full_pipeline_model.pkl')

score = pipeline.score(X_test, y_test)
print(f"Test accuracy: {score:.4f}")
