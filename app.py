import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title("🚢 Titanic Survival Prediction")
st.markdown("Predict whether a passenger would survive based on their details.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 1, 100, 25)
sibsp = st.slider("Number of Siblings/Spouses aboard", 0, 8, 0)
parch = st.slider("Number of Parents/Children aboard", 0, 6, 0)
fare = st.slider("Fare Paid", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

sex_val = 1 if sex == 'male' else 0
embarked_map = {'S': 0, 'C': 1, 'Q': 2}
embarked_val = embarked_map[embarked]


if st.button("Predict"):
    input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
    prediction = model.predict(input_data)
    result = "🎉 Survived" if prediction[0] == 1 else "❌ Did not survive"
    st.success(f"Prediction: {result}")
