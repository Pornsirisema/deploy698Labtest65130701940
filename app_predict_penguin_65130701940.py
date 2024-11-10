import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load model and encoders from the pickle file
@st.cache_resource  # Cache to avoid reloading every time the app is run
def load_model():
    with open('model_penguin_65130701940.pkl', 'rb') as file:
        model, species_encoder, island_encoder, sex_encoder = pickle.load(file)
    return model, species_encoder, island_encoder, sex_encoder

model, species_encoder, island_encoder, sex_encoder = load_model()

# Streamlit App
st.title("Penguin Species Classifier")

# Upload the user data
st.header("Enter the penguin features for prediction")
sex = st.selectbox('Select Sex', ['MALE', 'FEMALE'])
island = st.selectbox('Select Island', ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input('Culmen Length (mm)', min_value=0.0, max_value=100.0)
culmen_depth_mm = st.number_input('Culmen Depth (mm)', min_value=0.0, max_value=100.0)
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0, max_value=300.0)
body_mass_g = st.number_input('Body Mass (g)', min_value=0.0, max_value=10000.0)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'sex': [sex],
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g]
})

# Transform the categorical features using the encoders
sex_encoder.classes_
input_data['sex'] = sex_encoder.transform(input_data['sex'])
input_data['island'] = island_encoder.transform(input_data['island'])

# Make predictions
prediction = model.predict(input_data)

# Show prediction results
st.header("Prediction Result")
if prediction == 0:
    st.write("Predicted species: Adelie")
elif prediction == 1:
    st.write("Predicted species: Chinstrap")
else:
    st.write("Predicted species: Gentoo")
