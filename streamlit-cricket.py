import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

Match_Results = pd.read_csv('Match_Results.csv')

# Assuming 'Team 1', 'Team 2', 'Ground', and other relevant features as input features
features = ['Team 1', 'Team 2', 'Ground', 'Wins']

# Extracting features and target variable
X = Match_Results[features]
y = Match_Results['Winner']

# Convert categorical features into numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Assuming 'Team 1', 'Team 2', 'Ground', and other relevant features as input features
features = ['Team 1', 'Team 2', 'Ground', 'Wins']

# Streamlit app
st.title("ODI - Who Will Win?")

# Dropdown menu for selecting the ground
ground_columns = [col for col in X.columns if 'Ground' in col]
selected_ground = st.selectbox("Select Ground", ground_columns, index=0 if ground_columns else 0)

# Dropdown menu for selecting the countries playing
team1_columns = [col for col in X.columns if 'Team 1' in col]
selected_team_1 = st.selectbox("Select Team 1", team1_columns, index=0 if team1_columns else 0)

team2_columns = [col for col in X.columns if 'Team 2' in col]
selected_team_2 = st.selectbox("Select Team 2", team2_columns, index=0 if team2_columns else 0)

# Slider for Wins (you can replace this with your specific input features)
team1_wins = st.slider(f"{selected_team_1} Wins", min_value=0, max_value=10, value=5)
team2_wins = st.slider(f"{selected_team_2} Wins", min_value=0, max_value=10, value=5)

# Create the input data for prediction
input_data_dict = {
    'Team 1': [selected_team_1],
    'Team 2': [selected_team_2],
    'Ground': [selected_ground],
    'Wins': [team1_wins]
}

# Create the input_data DataFrame
input_data = pd.DataFrame(input_data_dict)

# Add a row for team2_wins (assuming both teams have the same number of wins)
input_data.loc[0, 'Wins_Team2'] = team2_wins

# Convert categorical features into numerical using one-hot encoding
input_data_encoded = pd.get_dummies(input_data)

# Ensure input_data_encoded has the same columns as X_train
missing_cols = set(X_train.columns) - set(input_data_encoded.columns)
for col in missing_cols:
    input_data_encoded[col] = 0

# Reorder columns to match X_train
input_data_encoded = input_data_encoded[X_train.columns]

# Make predictions using the loaded model
prediction = model.predict(input_data_encoded)[0]

# Display the predicted class
st.write(f"Predicted Winner: {prediction}")

