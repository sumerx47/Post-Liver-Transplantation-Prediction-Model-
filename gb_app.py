

import pandas as pd
import numpy as np
import pickle
import streamlit as st

pickle_in = open("gb_classifier.pkl","rb")
gb_classifier=pickle.load(pickle_in)



def liver_prediction(donor_age, donor_gender, donor_bmi, donor_cause_of_death, donor_diabetes, donor_hypertension, donor_alcohol_abuse, donor_smoking, donor_lymphocyte, donor_hepatitis_b, donor_hepatitis_c, recipient_etiology, recipient_meld_score, recipient_age, recipient_gender, recipient_bmi, recipient_diabetes, recipient_hypertension, recipient_alcohol_abuse, recipient_smoking, recipient_lymphocyte, recipient_hepatitis_b, recipient_hepatitis_c, recipient_albumin_level, recipient_alcoholic_cirrhosis, recipient_primary_biliary_cirrhosis, recipient_na, recipient_mg, recipient_wbc, recipient_platelets, recipient_cold_ischemia_time, recipient_warm_ischemia_time, recipient_blood_transfusion, recipient_immunosuppressant_medication, recipient_rejection_episodes):

   prediction = gb_classifier.predict([[donor_age, donor_gender, donor_bmi, donor_cause_of_death, donor_diabetes, donor_hypertension, donor_alcohol_abuse, donor_smoking, donor_lymphocyte, donor_hepatitis_b, donor_hepatitis_c, recipient_etiology, recipient_meld_score, recipient_age, recipient_gender, recipient_bmi, recipient_diabetes, recipient_hypertension, recipient_alcohol_abuse, recipient_smoking, recipient_lymphocyte, recipient_hepatitis_b, recipient_hepatitis_c, recipient_albumin_level, recipient_alcoholic_cirrhosis, recipient_primary_biliary_cirrhosis, recipient_na, recipient_mg, recipient_wbc, recipient_platelets, recipient_cold_ischemia_time, recipient_warm_ischemia_time, recipient_blood_transfusion, recipient_immunosuppressant_medication, recipient_rejection_episodes]])
   print(prediction)
   return prediction


def main():
    # giving a title
    st.title('Post Liver Transplantation Prediction Model')

    # getting the input data from the user
    donor_age = st.text_input('Donor Age (1-100)')
    donor_gender = st.text_input('Donor Gender (male = 1 , female = 0)')
    donor_bmi = st.text_input('Donor BMI (for eg: 30.9)')
    donor_cause_of_death = st.text_input('Donor Cause of Death (accident= 0 , other = 1 , stroke = 2)')
    donor_diabetes = st.text_input('Donor Diabetes (yes = 1 , No = 0)')
    donor_hypertension = st.text_input('Donor Hypertension (yes = 1 , No = 0)')
    donor_alcohol_abuse = st.text_input('Donor Alcohol Abuse (yes = 1 , No = 0)')
    donor_smoking = st.text_input('Donor Smoking (yes = 1 , No = 0)')
    donor_lymphocyte = st.text_input('Donor Lymphocyte (yes = 1 , No = 0)')
    donor_hepatitis_b = st.text_input('Donor Hepatitis B (yes = 1 , No = 0)')
    donor_hepatitis_c = st.text_input('Donor Hepatitis C (yes = 1 , No = 0)')
    recipient_etiology = st.text_input('Recipient Etiology (alcohol = 0 , hepatitis = 1 , nafld = 2 ,  other = 3)')
    recipient_meld_score = st.text_input('Recipient MELD Score (10-40)')
    recipient_age = st.text_input('Recipient Age (1-100)')
    recipient_gender = st.text_input('Recipient Gender  (male = 1 , female = 0)')
    recipient_bmi = st.text_input('Recipient BMI (for eg: 28.9)')
    recipient_diabetes = st.text_input('Recipient Diabetes (yes = 1 , No = 0)')
    recipient_hypertension = st.text_input('Recipient Hypertension (yes = 1 , No = 0)')
    recipient_alcohol_abuse = st.text_input('Recipient Alcohol Abuse (yes = 1 , No = 0)')
    recipient_smoking = st.text_input('Recipient Smoking (yes = 1 , No = 0)')
    recipient_lymphocyte = st.text_input('Recipient Lymphocyte (yes = 1 , No = 0)')
    recipient_hepatitis_b = st.text_input('Recipient Hepatitis B (yes = 1 , No = 0)')
    recipient_hepatitis_c = st.text_input('Recipient Hepatitis C (yes = 1 , No = 0)')
    recipient_albumin_level = st.text_input('Recipient Albumin Level (20-60)')
    recipient_alcoholic_cirrhosis = st.text_input('Recipient Alcoholic Cirrhosis (0-100)')
    recipient_primary_biliary_cirrhosis = st.text_input('Recipient Primary Biliary Cirrhosis (0.0-3.0)')
    recipient_na = st.text_input('Recipient Na (0.0-60.0)')
    recipient_mg = st.text_input('Recipient Mg (0.0-10.0)')
    recipient_wbc = st.text_input('Recipient WBC (2000-50000)')
    recipient_platelets = st.text_input('Recipient Platelets (20,000-5,00,000)')
    recipient_cold_ischemia_time = st.text_input('Recipient Cold Ischemia Time (0.0-20.0)')
    recipient_warm_ischemia_time = st.text_input('Recipient Warm Ischemia Time (0.0-20.0)')
    recipient_blood_transfusion = st.text_input('Recipient Blood Transfusion (0.0-20.0)')
    recipient_immunosuppressant_medication = st.text_input('Recipient Immunosuppressant Medication (cyclosporine = 0 , other = 1 , tacrolimus = 2)')
    recipient_rejection_episodes = st.text_input('Recipient Rejection Episodes (0-10)')

    # code for Prediction
    complication = ''
# creating a button for Prediction
    if st.button('Result'):
        complication_index = liver_prediction(int(donor_age), int(donor_gender), float(donor_bmi),
                                              int(donor_cause_of_death), int(donor_diabetes),
                                              int(donor_hypertension), int(donor_alcohol_abuse),
                                              int(donor_smoking), int(donor_lymphocyte),
                                              int(donor_hepatitis_b), int(donor_hepatitis_c),
                                              int(recipient_etiology), float(recipient_meld_score),
                                              int(recipient_age), int(recipient_gender), float(recipient_bmi),
                                              int(recipient_diabetes), int(recipient_hypertension),
                                              int(recipient_alcohol_abuse), int(recipient_smoking),
                                              float(recipient_lymphocyte), int(recipient_hepatitis_b),
                                              int(recipient_hepatitis_c), float(recipient_albumin_level),
                                              float(recipient_alcoholic_cirrhosis),
                                              float(recipient_primary_biliary_cirrhosis), float(recipient_na),
                                              float(recipient_mg), int(recipient_wbc), int(recipient_platelets),
                                              float(recipient_cold_ischemia_time),
                                              float(recipient_warm_ischemia_time),
                                              float(recipient_blood_transfusion),
                                              int(recipient_immunosuppressant_medication),
                                              int(recipient_rejection_episodes))

        if complication_index.size > 0:
            if complication_index[0] == 0:
                complication = '" Artery Thrombosis "'
            elif complication_index[0] == 1:
                complication = '" Biliary Complications "'
            elif complication_index[0] == 2:
                complication = '" Cardiovascular Complications "'
            elif complication_index[0] == 3:
                complication = '" Infection "'
            elif complication_index[0] == 4:
                complication = '" Metabolic Complications "'
            elif complication_index[0] == 5:
                complication = '" No Complication "'
            elif complication_index[0] == 6:
                complication = '" Portal Vein Thrombosis "'
            elif complication_index[0] == 7:
                complication = '" Post-transplant Diabetes "'
            elif complication_index[0] == 8:
                complication = '" Primary Graft Non-function "'
            elif complication_index[0] == 9:
                complication = '" Rejection "'
            elif complication_index[0] == 10:
                complication = '" Renal Dysfunction "'
            else:
                complication = 'Invalid prediction'

    st.success('Prediction result :  {}'.format(complication))

if __name__ == '__main__':
    main()
