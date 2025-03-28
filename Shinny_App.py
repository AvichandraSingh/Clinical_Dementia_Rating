import streamlit as st
import pandas as pd
import pickle

# Title and description of the app
st.title("CDR Prediction App")
st.write("Enter patient features to predict the Clinical Dementia Rating (CDR).")

# Load the trained model
@st.cache_resource
def load_model():
    with open("CDR_trained_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

model = load_model()

# Sidebar for user input
st.sidebar.header("Patient Input Parameters")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=50, max_value=100, value=65)
    gender = st.sidebar.selectbox("Gender (0: Male, 1: Female)", (0, 1))
    mmse = st.sidebar.number_input("MMSE Score", min_value=0, max_value=30, value=25)
    immediate = st.sidebar.number_input("Immediate Recall", min_value=0, max_value=10, value=5)
    delay = st.sidebar.number_input("Delayed Recall", min_value=0, max_value=10, value=4)
    vision_left = st.sidebar.number_input("Vision Score (Left)", min_value=0, max_value=5, value=1)
    audio_left = st.sidebar.number_input("Audio Score (Left)", min_value=0, max_value=5, value=1)
    mix_left = st.sidebar.number_input("Mix Score (Left)", min_value=0, max_value=5, value=1)
    flip_times = st.sidebar.number_input("Flip Times", min_value=0, max_value=10, value=2)
    flip_time = st.sidebar.number_input("Flip Time (s)", min_value=0.0, max_value=10.0, value=1.5)
    mouse_time = st.sidebar.number_input("Mouse Response Time", min_value=0.0, max_value=10.0, value=2.0)
    angle_degree = st.sidebar.number_input("Angle Degree", min_value=0, max_value=180, value=30)
    angle_time = st.sidebar.number_input("Angle Time", min_value=0.0, max_value=10.0, value=1.2)
    spiral_tremble = st.sidebar.number_input("Spiral Tremble", min_value=0.0, max_value=10.0, value=0.5)
    spiral_total_time = st.sidebar.number_input("Spiral Total Time", min_value=0.0, max_value=10.0, value=3.0)
    yong_time = st.sidebar.number_input("Yong Time", min_value=0.0, max_value=10.0, value=2.1)
    yong_tremble = st.sidebar.number_input("Yong Tremble", min_value=0.0, max_value=10.0, value=0.4)
    yong_overtimes = st.sidebar.number_input("Yong Overtimes", min_value=0, max_value=10, value=1)
    # Create a dictionary of user input
    data = {
        'Age': age,
        'Gender': gender,
        'MMSE': mmse,
        'immediate': immediate,
        'DELAY': delay,
        'Vision_Left': vision_left,
        'Audio_Left': audio_left,
        'Mix_Left': mix_left,
        'Flip_times': flip_times,
        'Flip_time': flip_time,
        'Mouse_time': mouse_time,
        'Angle_degree': angle_degree,
        'Angle_time': angle_time,
        'Spiral_tremble': spiral_tremble,
        'Spiral_totaltime': spiral_total_time,
        'Yong_time': yong_time,
        'Yong_tremble': yong_tremble,
        'Yong_overtimes': yong_overtimes,
    }

    
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display input data
st.subheader("Patient Input Parameters")
st.write(input_df)

# Predict using the model
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.subheader("Prediction Result")
    st.write(f"Predicted CDR Score: {prediction[0]}")