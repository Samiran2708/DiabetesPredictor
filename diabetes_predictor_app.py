import streamlit as st
import numpy as np
import pickle

# Load the trained model outside of the main function for efficiency
pickle_in = open("C:\\Users\\ghosh\\final_model.pkl", "rb")
final_model = pickle.load(pickle_in)

pickle_in_scaler = open("C:\\Users\\ghosh\\scaler.pkl", "rb")  # Assuming the scaler is saved as 'scaler.pkl'
scaler = pickle.load(pickle_in_scaler)

def predict_diabetes(pregnancies, glucose, insulin, bmi, dpf, age):
    
    input_data = np.array([[pregnancies, glucose, insulin, bmi, dpf, age]])

    scaled_data = scaler.transform(input_data)
    
    prediction = final_model.predict(scaled_data)
    prediction_proba = final_model.predict_proba(scaled_data)[0][1] 
    
    return prediction, prediction_proba

def main():
    st.title("Diabetes Prediction App")
   
    html_temp = """
    <h2 style="color:white; text-align:center;">Streamlit Diabetes Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields for the model
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=846, value=79)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    
    result = ""
    
    # Prediction logic
    if st.button("Submit"):
        prediction, prediction_proba = predict_diabetes(pregnancies, glucose, insulin, bmi, dpf, age)
        if prediction == [0]:
            result = "The prediction shows that you are not currently diabetic. Keep up with your healthy habits and regular checkups."
        elif prediction == [1]:
            result = "Based on the provided information, the model predicts that you may have diabetes, according to the Lehmann term. It is important to consult a healthcare professional for further assessment and guidance."
        st.success(f"The model predicts: {result}")
        st.write(f"Based on the information, the model estimates the probability of being diabetic as {prediction_proba * 100:.2f}%")
    
    # About section
    if st.button("About"):
            st.text("Welcome to the Diabetes Prediction App!")
            st.text("This app uses machine learning model (specifically Random Forest) to predict the likelihood of diabetes based on user inputs.")
            st.text("It is built with Streamlit to provide an interactive and user-friendly experience.")
            st.text("Remember, the model's prediction is just an estimate. Always consult a healthcare professional for a thorough assessment.")
            st.text("Built with Streamlit and Python.")

if __name__ == '__main__':
    main()



