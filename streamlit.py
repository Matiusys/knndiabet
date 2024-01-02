import pickle
import streamlit as st

    model = pickle.load(open('diabetes_model.sav', 'rb'))

st.set_page_config(
page_title="Prediksi Diabetes UAS",
page_icon="",
)

st.markdown(
"""
# Prediksi Penyakit Diabetes
  Prediksi Penyakit Diabetes bisa dilakukan dengan menginputkan beberapan data dibawah ini diantaranya :
"""
)


col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", 11,112,25)
with col2: 
    Pregnancies = st.number_input("Pregnancies", 13,130,29)

with col1:
    Insulin = st.number_input("Insulin", 120,210,175)
with col2: 
    Glucose = st.number_input("Glucose", 10,120,80)

# rumus bmi, digunakan sebagai default value
bmi_result = round((Glucose / ((Insulin / 100)**2)),2)
# Menghindari error min max value pada inputan
if(bmi_result <= 3.9) : bmi_result = 3.9
if(bmi_result >= 37.2)  : bmi_result = 37.2

BMI = st.number_input("Body mass index (Glucose / (Insulin)²))", 3.9,37.2,bmi_result

if st.button("Prediksi penyakit sakit jantung"):
    heart_disease_predict = model.predict([[Age,Glucose,Insulin,Pregnancies,DiabetesPedigreeFunction,BloodPressure,SkinThickness,BMI,outcome]])
    if(heart_disease_predict[0]==0):
        st.success("Pasien tidak terindikasi Penyakit diabetes")
    else :
        st.warning("Pasien terindikasi Penyakit diabetes")

