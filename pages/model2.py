import streamlit as st


st.subheader("credits")
st.write("https://archive.ics.uci.edu/dataset/45/heart+disease")
st.write("Thanks to ***V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:  Robert Detrano, M.D., Ph.D.***")

import streamlit as st

# Title
st.write("### 1. Data Features")

# Feature descriptions in markdown format
st.write(
  """
    ```age```: Age of the patient in years
    
    ```sex```: Sex (1 = male; 0 = female)
    
    ```cp```: Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)
    
    ```trestbps```: Resting blood pressure (in mm Hg on admission to the hospital)
    
    ```chol```: Serum cholestoral in mg/dl
    
    ```fbs```: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    
    ```restecg```: Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
    
    ```thalach```: Maximum heart rate achieved
    
    ```exang```: Exercise induced angina (1 = yes; 0 = no)
    
    ```oldpeak```: ST depression induced by exercise relative to rest
    
    ```slope```: The slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)
    
    ```ca```: Number of major vessels (0-3) colored by flourosopy
    
    ```thal```: 0 = normal; 1 = fixed defect; 2 = reversible defect
    
    ```num```: Diagnosis of heart disease (angiographic disease status 
         0 = < 50% diameter narrowing, 
         1 = > 50% diameter narrowing)
  """
)

# Explain the algorithms
st.write("### 2. Algorithm Theory")

st.write("""
- **K-Nearest Neighbors (KNN)**: A simple algorithm that classifies based on the majority label of the nearest neighbors.

- **Decision Tree**: A tree-like model that splits data based on feature values, and makes decisions based on the splits.
""")

st.write("### 3. Data Preparation")

def data_prep():
    st.write("- ##### 1) Loading the Data")
    st.write("The dataset is loaded from a CSV file (`processed.cleveland.csv`). Missing values represented by '?' are handled appropriately.")
    st.write("``` df = pd.read_csv('pages/processed.cleveland.csv', na_values='?') ```")

  # Display the first 5 rows
    st.write("- ##### 2). Display First 5 Rows")
    st.write("Once the dataset is loaded, we display the first 5 rows to understand the structure of the data and get an overview of the columns.")
    st.write("``` st.write(df.head())```")
    
    st.write("- ##### 3). Check Missing Values")
    st.write("Next, we check for missing values in the dataset. This is important because missing values can affect the performance of machine learning models.")
    st.write("``` missing_values = df.isnull().sum() ```") 
    st.write("``` missing_values ```")
    
    
    st.write("- ##### 4) Handling Missing Values")
    st.write("We fill missing values in the `ca` and `thal` columns using their median values to maintain data consistency.")
    st.write("``` df['ca'].fillna(df['ca'].median(), inplace=True) ```")
    st.write("``` df['thal'].fillna(df['thal'].median(), inplace=True) ```")

    
    st.write("- ##### 5) Feature and Target Separation")
    st.write("We separate the features (independent variables) and the target (`num`), which indicates the presence of heart disease.")
    st.write("``` X = df.drop('num', axis=1) ```")
    st.write("``` y = df['num'] ```")
    
    
    st.write("- ##### 6) Data Normalization")
    st.write("To improve model performance, we scale all numerical features using StandardScaler.")
    st.write("``` scaler = StandardScaler() ```")
    st.write("``` X_scaled = scaler.fit_transform(X) ```")
    
    
    st.write("- #### 7) Train-Test Split")
    st.write("We split the dataset into training (80%) and testing (20%) sets to evaluate model performance.")
    st.write("``` X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) ```")
data_prep()

