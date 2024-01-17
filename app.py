import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the pre-trained Random Forest Classifier model
with open('random_forest_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Streamlit UI
st.title("Titanic Survival Prediction")
st.sidebar.header("Enter Passenger Details")

# Input features
pclass = st.sidebar.selectbox("Pclass (Ticket class)", [1, 2, 3])
sex = st.sidebar.radio("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.slider("SibSp (Number of Siblings/Spouses Aboard)", 0, 8, 0)
parch = st.sidebar.slider("Parch (Number of Parents/Children Aboard)", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0, 512, 50)
embarked = st.sidebar.radio("Embarked", ["C", "Q", "S"])
family_size = sibsp + parch

# Display entered values
st.write("### Passenger Details:")
details = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked],
    "FamilySize": [family_size]
})
st.table(details)

# Preprocess input data
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked],
    "FamilySize": [family_size]
})

# Encode categorical variables
label_encoder = LabelEncoder()
input_data['Sex'] = label_encoder.fit_transform(input_data['Sex'])
input_data['Embarked'] = label_encoder.fit_transform(input_data['Embarked'])

# Make prediction
prediction = rf_model.predict(input_data)
prediction_proba = rf_model.predict_proba(input_data)

# Debugging information
st.write("### Debugging Information:")
st.write(f"Raw Predicted Probabilities: {prediction_proba}")

# Display prediction result
if prediction[0] == 1:
    st.write("### Prediction: The passenger is likely to survive.")
else:
    st.write("### Prediction: The passenger is unlikely to survive.")