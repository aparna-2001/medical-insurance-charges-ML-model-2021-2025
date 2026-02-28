import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_models():
    with open('gbm_model.pkl', 'rb') as f:
        gbm = pickle.load(f)
    with open('rf_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    return gbm, clf

gbm_model, clf_model = load_models()

st.set_page_config(
    page_title='Medical Insurance Predictor',
    page_icon='🏥',
    layout='centered'
)

st.title('🏥 Medical Insurance Predictor')
st.markdown('Predict insurance charges and detect smoker risk using Machine Learning')
st.divider()

st.sidebar.header('👤 Patient Information')

age = st.sidebar.slider('Age', min_value=18, max_value=65, value=30)
bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=60.0, value=28.0, step=0.1)
children = st.sidebar.selectbox('Number of Children', options=[0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox('Smoker', options=['No', 'Yes'])
sex = st.sidebar.selectbox('Sex', options=['Female', 'Male'])
region = st.sidebar.selectbox('Region', options=['Northeast', 'Northwest', 'Southeast', 'Southwest'])

mean_age = 39.22

smoker_yes            = 1 if smoker == 'Yes' else 0
sex_male              = 1 if sex == 'Male' else 0
region_northwest      = 1 if region == 'Northwest' else 0
region_southeast      = 1 if region == 'Southeast' else 0
region_southwest      = 1 if region == 'Southwest' else 0
age_smoker_interaction = (age - mean_age) * smoker_yes

input_data = pd.DataFrame({
    'age'                   : [age],
    'bmi'                   : [bmi],
    'children'              : [children],
    'sex_male'              : [sex_male],
    'smoker_yes'            : [smoker_yes],
    'region_northwest'      : [region_northwest],
    'region_southeast'      : [region_southeast],
    'region_southwest'      : [region_southwest],
    'age_smoker_interaction': [age_smoker_interaction]
})

st.subheader('📋 Patient Summary')
col1, col2, col3 = st.columns(3)
col1.metric('Age', age)
col2.metric('BMI', bmi)
col3.metric('Smoker', smoker)

st.divider()

if st.button('🔍 Predict', use_container_width=True):

    predicted_charge = gbm_model.predict(input_data)[0]

    st.subheader('💰 Predicted Insurance Charge')
    st.metric(
        label='Annual Charge Estimate',
        value=f'${predicted_charge:,.2f}',
        delta=f'${predicted_charge/12:,.2f} / month'
    )

    if predicted_charge < 5000:
        risk = '🟢 Low Risk'
        color = 'success'
    elif predicted_charge < 20000:
        risk = '🟡 Medium Risk'
        color = 'warning'
    else:
        risk = '🔴 High Risk'
        color = 'error'

    getattr(st, color)(f'Risk Level: {risk}')

    st.divider()

    st.subheader('🚬 Smoker Detection')

    clf_input = pd.DataFrame({
        'age'             : [age],
        'bmi'             : [bmi],
        'children'        : [children],
        'charges'         : [predicted_charge],
        'sex_male'        : [sex_male],
        'region_northwest': [region_northwest],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest]
    })

    smoker_proba = clf_model.predict_proba(clf_input)[0][1]
    smoker_pred  = clf_model.predict(clf_input)[0]

    col1, col2 = st.columns(2)
    col1.metric('Smoker Probability', f'{smoker_proba*100:.1f}%')
    col2.metric('Prediction', 'Smoker' if smoker_pred == 1 else 'Non-Smoker')

    if smoker_proba > 0.7:
        st.error('⚠️ High probability of smoker — flag for review')
    elif smoker_proba > 0.4:
        st.warning('⚠️ Moderate smoker probability — verify status')
    else:
        st.success('✅ Low smoker probability')

    st.divider()

    st.subheader('📊 Key Factors')
    factors = pd.DataFrame({
        'Factor'     : ['Smoker Status', 'BMI', 'Age'],
        'Your Value' : [smoker, bmi, age],
        'Impact'     : ['61%', '21%', '12%']
    })
    st.dataframe(factors, use_container_width=True, hide_index=True)
