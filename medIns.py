import pickle
import numpy as np
import streamlit as st

med_model = pickle.load(open('medInsurance.sav','rb'))

def med_insurance_prediction(input_data):
    
    input_data_as_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_array.reshape(1,-1)

    
    prediction = med_model.predict(input_data_reshaped)

    print('pred: ',prediction)

    return prediction[0]

def main():

    st.title("Medical Insurance Prediction")

    feature_explanations = {
        'Sex': ' 1: Male, 0: Female',
        'Smoker': '1: Yes, 0: No',
        'Region': '0: Southeast, 1: Southwest 2: Northeast, 3: Northwest'
    }
    
    # getting data from user
    age = int(st.number_input('Age', min_value=10, max_value=100, step=1, value=10))
    sex = int(st.number_input('Sex', min_value=0, max_value=1, step=1, value=0, help=feature_explanations['Sex']))
    bmi = float(st.number_input('BMI', min_value=10.00, max_value=60.00, step=0.10, value=10.00))
    children = int(st.number_input('Children', min_value=0, max_value=8, step=1, value=0))
    smoker = int(st.number_input('Smoking', min_value=0, max_value=1, step=1, value=0, help=feature_explanations['Smoker']))
    region = int(st.number_input('Region', min_value=0, max_value=3, step=1, value=0, help=feature_explanations['Region']))

    # code for prediction
    diagnosis = ''

    if st.button('Med Insurance Calc'):
        diagnosis = med_insurance_prediction([age,sex,bmi,children,smoker,region])
        diagnosis = f"The cost of Medical Insurance in USD is {diagnosis:.2f}"


    st.success(diagnosis)


if __name__ == '__main__':
    main()
