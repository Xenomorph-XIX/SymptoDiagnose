#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:52:44 2024

@author: hassaan
"""

import numpy as np
import pickle as p
import streamlit as st

loaded_model = p.load(open("prediction_model.sav", "rb"))

def diagnosis(input_data):
    
    input_data_as_np_array = np.asarray(input_data).reshape(1,-1)

    diagnosis = loaded_model.predict(input_data_as_np_array)[0]
    
    return f"The patient seems to have the following disease: {diagnosis}"


# Define a mapping of user-friendly symptom names to complex feature names
easier_to_complex_mapping = {
    "itching": "itching",
    "skin rash": "skin_rash",
    "nodal skin eruptions": "nodal_skin_eruptions",
    "continuous sneezing": "continuous_sneezing",
    "shivering": "shivering",
    "chills": "chills",
    "joint pain": "joint_pain",
    "stomach pain": "stomach_pain",
    "acidity": "acidity",
    "ulcers on tongue": "ulcers_on_tongue",
    "muscle wasting": "muscle_wasting",
    "vomiting": "vomiting",
    "burning micturition": "burning_micturition",
    "spotting urination": "spotting_urination",
    "fatigue": "fatigue",
    "weight gain": "weight_gain",
    "anxiety": "anxiety",
    "cold hands and feets": "cold_hands_and_feets",
    "mood swings": "mood_swings",
    "weight loss": "weight_loss",
    "restlessness": "restlessness",
    "lethargy": "lethargy",
    "patches in throat": "patches_in_throat",
    "irregular sugar level": "irregular_sugar_level",
    "cough": "cough",
    "high fever": "high_fever",
    "sunken eyes": "sunken_eyes",
    "breathlessness": "breathlessness",
    "sweating": "sweating",
    "dehydration": "dehydration",
    "indigestion": "indigestion",
    "headache": "headache",
    "yellowish skin": "yellowish_skin",
    "dark urine": "dark_urine",
    "nausea": "nausea",
    "loss of appetite": "loss_of_appetite",
    "pain behind the eyes": "pain_behind_the_eyes",
    "back pain": "back_pain",
    "constipation": "constipation",
    "abdominal pain": "abdominal_pain",
    "diarrhoea": "diarrhoea",
    "mild fever": "mild_fever",
    "yellow urine": "yellow_urine",
    "yellowing of eyes": "yellowing_of_eyes",
    "acute liver failure": "acute_liver_failure",
    "flud overload": "flud_overload",
    "swelling of stomach": "swelling_of_stomach",
    "swelled lymph nodes": "swelled_lymph_nodes",
    "malaise": "malaise",
    "blurred and distorted vision": "blurred_and_distorted_vision",
    "phlegm": "phlegm",
    "throat irritation": "throat_irritation",
    "redness of eyes": "redness_of_eyes",
    "sinus pressure": "sinus_pressure",
    "runny nose": "runny_nose",
    "congestion": "congestion",
    "chest pain": "chest_pain",
    "weakness in limbs": "weakness_in_limbs",
    "fast heart rate": "fast_heart_rate",
    "pain during bowel movements": "pain_during_bowel_movements",
    "pain in anal region": "pain_in_anal_region",
    "bloody stool": "bloody_stool",
    "irritation in anus": "irritation_in_anus",
    "neck pain": "neck_pain",
    "dizziness": "dizziness",
    "cramps": "cramps",
    "bruising": "bruising",
    "obesity": "obesity",
    "swollen legs": "swollen_legs",
    "swollen blood vessels": "swollen_blood_vessels",
    "puffy face and eyes": "puffy_face_and_eyes",
    "enlarged thyroid": "enlarged_thyroid",
    "brittle nails": "brittle_nails",
    "swollen extremeties": "swollen_extremeties",
    "excessive hunger": "excessive_hunger",
    "extra marital contacts": "extra_marital_contacts",
    "drying and tingling lips": "drying_and_tingling_lips",
    "slurred speech": "slurred_speech",
    "knee pain": "knee_pain",
    "hip joint pain": "hip_joint_pain",
    "muscle weakness": "muscle_weakness",
    "stiff neck": "stiff_neck",
    "swelling joints": "swelling_joints",
    "movement stiffness": "movement_stiffness",
    "spinning movements": "spinning_movements",
    "loss of balance": "loss_of_balance",
    "unsteadiness": "unsteadiness",
    "weakness of one body side": "weakness_of_one_body_side",
    "loss of smell": "loss_of_smell",
    "bladder discomfort": "bladder_discomfort",
    "foul smell of urine": "foul_smell_of_urine",
    "continuous feel of urine": "continuous_feel_of_urine",
    "passage of gases": "passage_of_gases",
    "internal itching": "internal_itching",
    "toxic look (typhos)": "toxic_look_(typhos)",
    "depression": "depression",
    "irritability": "irritability",
    "muscle pain": "muscle_pain",
    "altered sensorium": "altered_sensorium",
    "red spots over body": "red_spots_over_body",
    "belly pain": "belly_pain",
    "abnormal menstruation": "abnormal_menstruation",
    "dischromic patches": "dischromic_patches",
    "watering from eyes": "watering_from_eyes",
    "increased appetite": "increased_appetite",
    "polyuria": "polyuria",
    "family history": "family_history",
    "mucoid sputum": "mucoid_sputum",
    "rusty sputum": "rusty_sputum",
    "lack of concentration": "lack_of_concentration",
    "visual disturbances": "visual_disturbances",
    "receiving blood transfusion": "receiving_blood_transfusion",
    "receiving unsterile injections": "receiving_unsterile_injections",
    "coma": "coma",
    "stomach bleeding": "stomach_bleeding",
    "distention of abdomen": "distention_of_abdomen",
    "history of alcohol consumption": "history_of_alcohol_consumption",
    "fluid overload": "fluid_overload",
    "blood in sputum": "blood_in_sputum",
    "prominent veins on calf": "prominent_veins_on_calf",
    "palpitations": "palpitations",
    "painful walking": "painful_walking",
    "pus filled pimples": "pus_filled_pimples",
    "blackheads": "blackheads",
    "scurring": "scurring",
    "skin peeling": "skin_peeling",
    "silver like dusting": "silver_like_dusting",
    "small dents in nails": "small_dents_in_nails",
    "inflammatory nails": "inflammatory_nails",
    "blister": "blister",
    "red sore around nose": "red_sore_around_nose",
    "yellow crust ooze": "yellow_crust_ooze",
}

complex_to_index_mapping = {name: str(index) for index, name in enumerate([
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
    "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
    "vomiting", "burning_micturition", "spotting_urination", "fatigue", "weight_gain",
    "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
    "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
    "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes",
    "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "flud_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain",
    "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region",
    "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising",
    "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid",
    "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts",
    "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness",
    "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance",
    "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
    "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching",
    "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic_patches",
    "watering_from_eyes", "increased_appetite", "polyuria", "family_history", "mucoid_sputum",
    "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf", "palpitations",
    "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister",
    "red_sore_around_nose", "yellow_crust_ooze"
])}

available_symptoms = [
    'itching',
    'skin rash',
    'nodal skin eruptions',
    'continuous sneezing',
    'shivering',
    'chills',
    'joint pain',
    'stomach pain',
    'acidity',
    'ulcers on tongue',
    'muscle wasting',
    'vomiting',
    'burning micturition',
    'spotting urination',
    'fatigue',
    'weight gain',
    'anxiety',
    'cold hands and feets',
    'mood swings',
    'weight loss',
    'restlessness',
    'lethargy',
    'patches in throat',
    'irregular sugar level',
    'cough',
    'high fever',
    'sunken eyes',
    'breathlessness',
    'sweating',
    'dehydration',
    'indigestion',
    'headache',
    'yellowish skin',
    'dark urine',
    'nausea',
    'loss of appetite',
    'pain behind the eyes',
    'back pain',
    'constipation',
    'abdominal pain',
    'diarrhoea',
    'mild fever',
    'yellow urine',
    'yellowing of eyes',
    'acute liver failure',
    'flud overload',
    'swelling of stomach',
    'swelled lymph nodes',
    'malaise',
    'blurred and distorted vision',
    'phlegm',
    'throat irritation',
    'redness of eyes',
    'sinus pressure',
    'runny nose',
    'congestion',
    'chest pain',
    'weakness in limbs',
    'fast heart rate',
    'pain during bowel movements',
    'pain in anal region',
    'bloody stool',
    'irritation in anus',
    'neck pain',
    'dizziness',
    'cramps',
    'bruising',
    'obesity',
    'swollen legs',
    'swollen blood vessels',
    'puffy face and eyes',
    'enlarged thyroid',
    'brittle nails',
    'swollen extremities',
    'excessive hunger',
    'extra marital contacts',
    'drying and tingling lips',
    'slurred speech',
    'knee pain',
    'hip joint pain',
    'muscle weakness',
    'stiff neck',
    'swelling joints',
    'movement stiffness',
    'spinning movements',
    'loss of balance',
    'unsteadiness',
    'weakness of one body side',
    'loss of smell',
    'bladder discomfort',
    'foul smell of urine',
    'continuous feel of urine',
    'passage of gases',
    'internal itching',
    'toxic look (typhos)',
    'depression',
    'irritability',
    'muscle pain',
    'altered sensorium',
    'red spots over body',
    'belly pain',
    'abnormal menstruation',
    'dischromic patches',
    'watering from eyes',
    'increased appetite',
    'polyuria',
    'family history',
    'mucoid sputum',
    'rusty sputum',
    'lack of concentration',
    'visual disturbances',
    'receiving blood transfusion',
    'receiving unsterile injections',
    'coma',
    'stomach bleeding',
    'distention of abdomen',
    'history of alcohol consumption',
    'fluid overload',
    'blood in sputum',
    'prominent veins on calf',
    'palpitations',
    'painful walking',
    'pus filled pimples',
    'blackheads',
    'scurring',
    'skin peeling',
    'silver like dusting',
    'small dents in nails',
    'inflammatory nails',
    'blister',
    'red sore around nose',
    'yellow crust ooze'
]


def get_complex_symptom_name(easier_name): return easier_to_complex_mapping.get(easier_name, None)

def get_feature_index(complex_name): return complex_to_index_mapping.get(complex_name, None)


selected_symptoms = []


def get_feature_vector(symptoms):
    # Initialize a vector with zeros
    feature_vector = np.zeros(len(complex_to_index_mapping), dtype=int)
    
    for symptom in symptoms:
        complex_name = easier_to_complex_mapping.get(symptom)
        if complex_name:
            index = complex_to_index_mapping.get(complex_name)
            if index is not None:
                feature_vector[int(index)] = 1
    
    return feature_vector

st.title('SymptoDiagnose Web Application')

st.sidebar.title('Available Symptoms')
selected_symptoms = st.sidebar.multiselect(
    'Select symptoms to add to your list:',
    available_symptoms
)

# Initialize session state for symptoms list
if 'symptoms_list' not in st.session_state:
    st.session_state.symptoms_list = []

# Button to add selected symptoms to the symptoms list
if st.sidebar.button('Add Symptoms'):
    for symptom in selected_symptoms:
        if symptom not in st.session_state.symptoms_list:
            st.session_state.symptoms_list.append(symptom)

# Display current symptoms list


st.write("### Current Symptoms:")
st.write("Here are the symptoms you have added:")
for symptom in st.session_state.symptoms_list:
    st.write(f"- {symptom}")

# Diagnose button
if st.button('Diagnose'):
    
    feature_vector = get_feature_vector(st.session_state.symptoms_list)
    
    if np.all(feature_vector == 0): prediction = "No symptoms were selected"
        
    else: prediction = diagnosis(feature_vector)
    
    st.write(prediction)

# Clear symptoms list
if st.button('Clear Symptoms List'):
    st.session_state.symptoms_list = []
    st.write('Symptoms list cleared.')




