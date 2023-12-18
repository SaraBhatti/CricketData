import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load your batting data (replace 'batsman_data.csv' with the actual file path)
all_batters = pd.read_csv('all_batters.csv')

# Assuming 'Inns', 'Mat', 'Ave', 'SR' as input features
features = ['Inns', 'Mat', 'Ave', 'SR']

# Extracting features and target variable
X = all_batters[features]
y = all_batters['Runs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Streamlit app
st.title("How's many runs will you make?")

# Corrected background image URL
background_image_url = "https://static.vecteezy.com/system/resources/previews/021/916/215/large_2x/batsman-playing-cricket-championship-sports-free-photo.jpg"

background_image = f"""
<style>
body {{
    background-image: url('{background_image_url}');
    background-size: cover;
}}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# Create input features using Streamlit sliders
inns = st.slider("Innings Batted", min_value=0, max_value=50, value=25)
mat = st.slider("Matches", min_value=0, max_value=100, value=50)
ave = st.slider("Batting Average", min_value=0, max_value=100, value=50)
sr = st.slider("Strike Rate", min_value=0, max_value=200, value=100)

# Make predictions using the loaded Random Forest model
input_data_rf = np.array([[inns, mat, ave, sr]])
prediction_rf = model_rf.predict(input_data_rf)[0]

# Display the prediction
st.write(f"Predicted Runs: {prediction_rf}")
