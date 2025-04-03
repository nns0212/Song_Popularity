import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("C:\\Users\\nchpr\\PycharmProjects\\PythonProject\\.venv\\song_data.csv")
# Ensure correct path
        return df
    except FileNotFoundError:
        st.error("Error: 'song_data.csv' not found. Please check the file location.")
        return None

df = load_data()

# Check if dataset is loaded successfully
if df is not None:
    st.title("🎵 Song Popularity Prediction")
    st.subheader("Loading Dataset ✅")
    st.write(df.head())  # Show sample data

# Model Training (Only Runs Once and Doesn't Display Evaluation)
@st.cache_resource
def train_model(df):
    if df is None:
        return None

    # Define features and target
    features = [col for col in df.columns if col not in ["song_name", "song_popularity"]]
    X = df[features]
    y = df["song_popularity"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))  # Ridge Regression Model
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, "best_model.pkl")

    return model, features

model, features = train_model(df)

# Load trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Trained model not found! Please train and save the model first.")
        return None

model = load_model()

# Prediction Function
def predict_song_popularity(song_name, model, df, features):
    if df is None or model is None:
        return None

    if 'song_name' in df.columns and song_name in df['song_name'].values:
        song_data = df[df['song_name'] == song_name][features]
    else:
        st.warning(f"Song '{song_name}' not found in dataset. Please enter feature values manually.")
        return None

    X_song = song_data.values
    X_song = model.named_steps['scaler'].transform(X_song)  # Apply scaling
    prediction = model.named_steps['regressor'].predict(X_song)[0]
    return prediction

# Streamlit UI for User Input
if df is not None and model is not None:
    song_list = df["song_name"].unique().tolist()
    song_to_predict = st.selectbox("Select a song:", song_list)

    if st.button("Predict Popularity"):
        predicted_popularity = predict_song_popularity(song_to_predict, model, df, features)

        if predicted_popularity is not None:
            st.success(f"Predicted popularity for '{song_to_predict}': {predicted_popularity:.2f}")

