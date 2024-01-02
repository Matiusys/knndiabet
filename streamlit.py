import streamlit as st
import pickle

model = pickle.load(open('Diabetes_model.sav','rb'))

st.title("Klasifikasi Diabetes")


Age = st.number_input("Umur", 11,112,25)

Glucose = st.number_input("Masukan data Glucose", 0, 200, 0)

Insulin = st.number_input("Masukan data Insulin", 0, 100, 10)

Pregnancies = st.number_input("Masukan data Pregnancies", 0, 10, 0)

BloodPressure = st.number_input("Masukan data BloodPressure", 0, 120, 0)

BMI = st.number_input("Masukan data BMI", 0, 50, 0)

SkinThickness = st.number_input("Masukan data SkinThickness", 0, 50, 10)

DiabetesPedigreeFunction = st.number_input("Masukan data DiabetesPedigreeFunction", 0.0, 6.2, 3.2)

if st.button("Klasifikasi Diabetes"):
    diabetes_predict = model.predict([Age, Glucose, Insulin, Pregnancies, BloodPressure, BMI, SkinThickness, DiabetesPedigreeFunction, Outcome])
    if (diabetes_predict[0] == 0):
      st.success = 'Patients Affected by Diabetes '
    else :
      st.warning = 'Patient is not Affected by Diabetes'