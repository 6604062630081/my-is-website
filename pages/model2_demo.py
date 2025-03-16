import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def clean_data():
    df = pd.read_csv('pages/processed.cleveland.csv', na_values="?")
    
    df['num'] = (df['num'] > 0).astype(int)  # Convert target to binary classification
    df['ca'].fillna(df['ca'].median(), inplace=True)
    df['thal'].fillna(df['thal'].median(), inplace=True)
    
    X = df.drop('num', axis=1)
    y = df['num']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

def train_models():
    X_train, X_test, y_train, y_test, scaler = clean_data()
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    
    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    
    # st.write(f"KNN Accuracy: {knn_acc:.2f}")
    # st.write(f"Decision Tree Accuracy: {dt_acc:.2f}")
    
    return knn, dt, scaler

def user_input_form():
    st.title("Heart Disease Prediction")
    
    col1, col2 = st.columns(2)
    age = col1.number_input("Age", min_value=29, max_value=77, value=62)
    sex = 1 if col2.selectbox("Sex", ["Male", "Female"]) == "Male" else 0

    col3, col4 = st.columns(2)
    cp = col3.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
    trestbps = col4.number_input("Resting Blood Pressure", min_value=94, max_value=200, value=120)
    
    col5, col6 = st.columns(2)
    chol = col5.number_input("Serum Cholesterol (mg/dl)", min_value=126, max_value=564, value=260)
    fbs = 1 if col6.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"]) == "Yes" else 0
    col7, col8 = st.columns(2)
    restecg = col7.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=1)
    thalach = col8.number_input("Max Heart Rate Achieved", min_value=71, max_value=202, value=150)

    col9, col10 = st.columns(2)
    exang = 1 if col9.selectbox("Exercise Induced Angina", ["Yes", "No"]) == "Yes" else 0
    oldpeak = col10.number_input("ST Depression", min_value=0.0, max_value=6.2, value=1.0)

    col11, col12 = st.columns(2)
    slope = col11.number_input("Slope of ST Segment (0-2)", min_value=0, max_value=2, value=1)
    ca = col12.number_input("Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia (3,6,7)", ["3", "6", '7'], index=0)
    
    user_input = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    return user_input

def predict():
    knn, dt, scaler = train_models()
    user_input = user_input_form()
    user_input_scaled = scaler.transform([user_input])
    
    prediction_knn = knn.predict(user_input_scaled)[0]
    prediction_dt = dt.predict(user_input_scaled)[0]
    
    st.write("Prediction from KNN Model: ", " ***Heart Disease***" if prediction_knn else " ***No Heart Disease***")
    st.write("Prediction from Decision Tree Model: ", " ***Heart Disease***" if prediction_dt else " ***No Heart Disease***")

if __name__ == "__main__":
    predict()
