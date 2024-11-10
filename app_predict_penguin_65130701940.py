import pickle
import pandas as pd
import streamlit as st

# Load the pre-trained model and encoders
with open('model_penguin_65130701940.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Streamlit app header
st.title("Penguin Species Prediction")

# User input for penguin features
island = st.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.slider('Culmen Length (mm)', 30.0, 60.0, 37.0)
culmen_depth_mm = st.slider('Culmen Depth (mm)', 10.0, 25.0, 19.3)
flipper_length_mm = st.slider('Flipper Length (mm)', 150.0, 250.0, 192.3)
body_mass_g = st.slider('Body Mass (g)', 2500, 6000, 3750)
sex = st.selectbox('Sex', ['MALE', 'FEMALE'])

# Prepare the input data for prediction
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Transform categorical data using the encoders
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# Predict the species
y_pred_new = model.predict(x_new)

# Inverse transform the prediction to get the species name
result = species_encoder.inverse_transform(y_pred_new)

# Show the result in Streamlit
st.subheader('Predicted Species:')
st.write(result[0])

