import streamlit as st

st.header("Neural Network")


st.write("""A neural network is an artificial intelligence (AI) technique that enables computers to process data in a manner inspired by the human brain. As a subset of machine learning (ML), specifically deep learning, it consists of interconnected nodes or neurons arranged in multiple layers, mimicking the brain’s structure. This adaptive system allows computers to learn from errors and continuously enhance their performance. As a result, artificial neural networks are capable of solving complex problems, such as document summarization and facial recognition, with high accuracy.""")
st.subheader("Neural Network Learning Architecture")

image_url1 = "https://francescolelli.info/wp-content/uploads/2019/05/NeuralNetworks-input-layer-hidden-layer-output-layer.png"
st.image(image_url1, use_container_width=True)
st.write("""
Neural networks build upon traditional machine learning by introducing multiple layers of interconnected neurons, allowing them to learn hierarchical patterns in data. Unlike classical machine learning algorithms that rely on manually designed features, neural networks automatically extract and refine features through a process known as representation learning.

A neural network consists of four key components:

1. Input Layer – The input layer receives raw data, which is represented as numerical values (features). Each node in this layer corresponds to a specific feature from the dataset.

2. Hidden Layers – Hidden layers perform complex computations by transforming the input data into meaningful representations. Each neuron in a hidden layer processes inputs from the previous layer, applies weights, computes an activation function, and passes the result to the next layer. The number of hidden layers and neurons determines the network’s capacity to learn complex patterns.

3. Output Layer – The final layer of the network generates predictions based on the processed information. The number of neurons in this layer depends on the type of task:

    - For binary classification, the output layer typically has a single neuron with a Sigmoid activation function (outputting probabilities between 0 and 1).

    - For multiclass classification, it has multiple neurons, each corresponding to a class, with a Softmax activation function.

    - For regression tasks, it has a single neuron with a linear activation function (outputting continuous values).

4. Prediction – The output layer produces a final decision based on probabilities or scores. This prediction is interpreted and used for decision-making.
""")
st.subheader("Forward Propagation: How Neural Networks Process Data")
st.write("""The process of computing outputs from inputs is called forward propagation. This involves passing data sequentially through each layer:

1. Weighted Sum Calculation - Each neuron computes a weighted sum of its inputs:""")
st.code("""
z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b

""", language="python")

st.write("""where w represents the weights, x represents the inputs, and b is a bias term that helps improve flexibility.

2. Activation Function - The computed value  is passed through an activation function that introduces non-linearity, enabling the network to learn complex patterns. Common activation functions include:

    - **ReLU (Rectified Linear Unit)**: Used in hidden layers to introduce non-linearity.

    - **Sigmoid**: Used in binary classification to output probabilities.

    - **Softmax**: Used in multiclass classification to distribute probabilities among multiple classes.

3. Output Generation - The transformed values are passed through subsequent layers until the final output is generated.
""")

st.write("""ที่มาของ ข้อมูล: [aws](https://aws.amazon.com/what-is/neural-network/?nc1=h_ls), [Github](https://guopai.github.io/ml-blog14.html) | รูปภาพ: [francescolelli](https://francescolelli.info/tutorial/neural-networks-a-collection-of-youtube-videos-for-learning-the-basics/)""")



st.write(""" Dataset's source: https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being/data""")
st.write("""thanks to https://www.kaggle.com/code/emirhanai/predicting-emotional-well-being-from-social-media""")


st.write("### 1. Data Features")

# Feature descriptions in markdown format
st.write(
  """
    ```User_ID```: Unique identifier for the user.

    ```Age```: Age of the user.

    ```Gender```: Gender of the user (Female, Male, Non-binary).

    ```Platform```: Social media platform used (e.g., Instagram, Twitter, Facebook, LinkedIn, Snapchat, Whatsapp, Telegram).

    ```Daily_Usage_Time (minutes)```: Daily time spent on the platform in minutes.

    ```Posts_Per_Day```: Number of posts made per day.

    ```Likes_Received_Per_Day```: Number of likes received per day.

    ```Comments_Received_Per_Day```: Number of comments received per day.

    ```Messages_Sent_Per_Day```: Number of messages sent per day.

    ```Dominant_Emotion```: User's dominant emotional state during the day (e.g., Happiness, Sadness, Anger, Anxiety, Boredom, Neutral).

  """
)

st.write("### 2. Data Preparation")
st.write("""
    **Data Cleaning and Preparation:**

    1. **Loading the dataset:**
        - The dataset is loaded from CSV files using `pd.read_csv()`.
        - Three datasets are loaded: `train.csv`, `test.csv`, and `val.csv` for training, testing, and validation, respectively.

    2. **Handling missing values:**
        - Missing values in the dataset are handled by dropping rows with missing values using `dropna()`.
        - This step ensures that the data used for training, testing, and validation is clean and free from any missing values.

    3. **Encoding categorical features:**
        - The categorical columns `'Gender'` and `'Platform'` are one-hot encoded using `pd.get_dummies()`.
        - This step transforms categorical values into binary variables (dummy variables), allowing the model to process them properly.
        - The `drop_first=True` argument ensures that the first level of each category is dropped to avoid multicollinearity.

    4. **Separating features (X) and target (y):**
        - The target column, `'Dominant_Emotion'`, is separated from the feature columns.
        - The features (`X`) are everything except the target, and the target (`y`) is the `'Dominant_Emotion'` column.

    5. **Scaling numeric features:**
        - The numeric columns are selected and scaled using `StandardScaler()`.
        - `StandardScaler()` standardizes the data by removing the mean and scaling it to unit variance.
        - The training data is fitted using `scaler.fit_transform()`, while the testing and validation data are transformed using `scaler.transform()`.

    6. **Encoding the target labels:**
        - The target labels (`Dominant_Emotion`) are encoded using `LabelEncoder()`.
        - This converts the target variable into numeric labels.
        - Then, the numeric labels are converted into one-hot encoded vectors using `to_categorical()` to represent each emotion as a one-hot vector.

    7. **Preparing data for GRU input:**
        - The data is reshaped for use in the GRU model by adding an extra dimension using `np.expand_dims()`.
        - This reshaping is necessary as GRU models expect 3D input with the shape `(samples, timesteps, features)`.

    **Summary of Data Preparation Process:**
    - Dropped missing values.
    - One-hot encoded categorical variables.
    - Scaled numerical features.
    - Encoded the target labels.
    - Reshaped the data for the GRU model.
""")
