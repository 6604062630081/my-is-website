import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.utils import to_categorical

def model_1():
    # Load the dataset
    train_df = pd.read_csv('pages/train.csv', on_bad_lines='skip')
    test_df = pd.read_csv('pages/test.csv', on_bad_lines='skip')
    val_df = pd.read_csv('pages/val.csv', on_bad_lines='skip')

    # # Show the first 5 rows of the training data
    # st.write("First 5 rows of the training data:")
    # st.write(train_df.head())

    # # Display missing values count
    # st.write("\nMissing values in the training data:")
    # st.write(train_df.isnull().sum())

    # Handle missing values by dropping rows
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    val_df = val_df.dropna()

    # Ensure categorical columns are encoded before scaling
    train_df = pd.get_dummies(train_df, columns=['Gender', 'Platform'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Gender', 'Platform'], drop_first=True)
    val_df = pd.get_dummies(val_df, columns=['Gender', 'Platform'], drop_first=True)

    # Now, separate features (X) and target (y) columns
    X_train = train_df.drop('Dominant_Emotion', axis=1)
    X_test = test_df.drop('Dominant_Emotion', axis=1)
    X_val = val_df.drop('Dominant_Emotion', axis=1)

    # Only select numeric columns for scaling
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Apply StandardScaler to only numeric columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled = scaler.transform(X_test[numeric_columns])
    X_val_scaled = scaler.transform(X_val[numeric_columns])

    # Combine all labels from train, test, and validation sets for LabelEncoder
    all_labels = pd.concat([train_df['Dominant_Emotion'], test_df['Dominant_Emotion'], val_df['Dominant_Emotion']])

    # One-hot encode the target variable (Dominant_Emotion)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)  # Fit on all labels from train, test, and validation

    y_train_encoded = label_encoder.transform(train_df['Dominant_Emotion'])
    y_test_encoded = label_encoder.transform(test_df['Dominant_Emotion'])
    y_val_encoded = label_encoder.transform(val_df['Dominant_Emotion'])

    # Convert the encoded labels into one-hot encoding
    y_train_one_hot = to_categorical(y_train_encoded)
    y_test_one_hot = to_categorical(y_test_encoded)
    y_val_one_hot = to_categorical(y_val_encoded)

    # Prepare data for GRU (3D input for GRU)
    X_train_gru = np.expand_dims(X_train_scaled, axis=1)
    X_test_gru = np.expand_dims(X_test_scaled, axis=1)
    X_val_gru = np.expand_dims(X_val_scaled, axis=1)

    # Build the GRU model
    model = Sequential()
    model.add(GRU(32, activation="relu", return_sequences=True, input_shape=(1, X_train_gru.shape[2])))
    model.add(Dropout(0.2))

    model.add(GRU(128, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(64, activation='tanh', return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="tanh"))
    model.add(Dropout(0.2))

    output_classes = y_train_one_hot.shape[1]
    model.add(Dense(output_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model when button is clicked
    if st.button('Train the Model'):
        history = model.fit(X_train_gru, y_train_one_hot, epochs=300, batch_size=32, validation_data=(X_val_gru, y_val_one_hot))
        st.write("Model Training Complete")

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test_gru, y_test_one_hot)
        st.write(f"Test Loss: {test_loss}")
        st.write(f"Test Accuracy: {test_accuracy}")

    # User input for prediction
    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Age", max_value=100, value=25)
    gender = col2.selectbox("Gender", ["Male", "Female", "Non-binary"])
    platform = col3.selectbox("Platform", ["Instagram", "Twitter", "Facebook", "Snapchat", "Linkedin", "Whatsapp", "Telegram"])
    col4, col5 = st.columns(2)
    
    feature_1 = col4.number_input("Daily_Usage_Time (minutes)", min_value=0)
    feature_2 = col5.number_input("Posts_Per_Day", min_value=0)
    col6, col7, col8 = st.columns(3)
    feature_3 = col6.number_input("Likes_Received_Per_Day", min_value=0)
    feature_4 = col7.number_input("Comments_Received_Per_Day", min_value=0)
    feature_5 = col8.number_input("Messages_Sent_Per_Day", min_value=0)

    # Create the user input DataFrame
    user_input = pd.DataFrame({
        'Gender': [gender],
        'Platform': [platform],
        'Age': [age],
        'Daily_Usage_Time (minutes)': [feature_1],
        'Posts_Per_Day': [feature_2],
        'Likes_Received_Per_Day': [feature_3],
        'Comments_Received_Per_Day': [feature_4],
        'Messages_Sent_Per_Day': [feature_5]
    })

    # One-hot encode the categorical variables for user input
    user_input = pd.get_dummies(user_input, columns=['Gender', 'Platform'], drop_first=True)

    # Scale the numeric features for user input
    user_input_scaled = scaler.transform(user_input[numeric_columns])

    # Reshape the input for GRU (3D input)
    user_input_gru = np.expand_dims(user_input_scaled, axis=1)

    # Make prediction with the trained model
    if st.button('Predict Emotion'):
        prediction = model.predict(user_input_gru)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_emotion = label_encoder.inverse_transform(predicted_class)
        st.write(f"Predicted Emotion: {predicted_emotion[0]}")

# Run Streamlit app
if __name__ == "__main__":
    st.title("Emotion Detection GRU Model Demo")
    model_1()
