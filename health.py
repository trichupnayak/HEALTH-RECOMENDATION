import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

# Define directories
MODEL_DIR = Path(r"C:\Users\preet\OneDrive\Semester 4\INTERN")
DATA_DIR = Path(r"C:\Users\preet\OneDrive\Semester 4\INTERN\Datasets")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Define mappings
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
    'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
    'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13,
    'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
    'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23,
    'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29,
    'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34,
    'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39,
    'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48,
    'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52,
    'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57,
    'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67,
    'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
    'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79,
    'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83,
    'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87,
    'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91,
    'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95,
    'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
    'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130,
    'yellow_crust_ooze': 131
}
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
    25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal Positional Vertigo',
    2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# Helper functions
@st.cache_data
def load_recommendation_data():
    try:
        datasets = {
            'symtoms': pd.read_csv(DATA_DIR / "C:\\Users\\preet\\OneDrive\\Semester  4\\INTERN\\datasets\\symtoms_df.csv"),  # Corrected typo
            'precautions': pd.read_csv(DATA_DIR / "C:\\Users\\preet\\OneDrive\\Semester  4\\INTERN\\datasets\\precautions_df.csv"),
            'workouts': pd.read_csv(DATA_DIR / "C:\\Users\\preet\\OneDrive\\Semester  4\\INTERN\\datasets\\workout_df.csv"),
            'descriptions': pd.read_csv(DATA_DIR / "C:\\Users\\preet\\OneDrive\\Semester  4\\INTERN\\datasets\\description.csv"),
            'medications': pd.read_csv(DATA_DIR / "C:\\Users\\preet\\OneDrive\\Semester  4\\INTERN\\datasets\\medications.csv"),
            'diets': pd.read_csv(DATA_DIR / "C:\\Users\\preet\\OneDrive\\Semester  4\\INTERN\\datasets\\diets.csv")
        }
        return datasets
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Ensure all datasets are in {DATA_DIR}.")
        return None

def helper(dis, datasets):
    desc = datasets['descriptions'][datasets['descriptions']['Disease'] == dis]['Description']
    desc = " ".join(desc) if not desc.empty else "No description available"
    pre = datasets['precautions'][datasets['precautions']['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values.flatten() if pd.notna(col)] if not pre.empty else ["No precautions available"]
    med = datasets['medications'][datasets['medications']['Disease'] == dis]['Medication']
    med = [m for m in med.values] if not med.empty else ["No medications available"]
    die = datasets['diets'][datasets['diets']['Disease'] == dis]['Diet']
    die = [d for d in die.values] if not die.empty else ["No diet recommendations available"]
    wrkout = datasets['workouts'][datasets['workouts']['disease'] == dis]['workout']
    wrkout = [w for w in wrkout.values] if not wrkout.empty else ["No workout recommendations available"]
    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms, model):
    input_vector = np.zeros(len(symptoms_dict))
    unrecognized = []
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
        else:
            unrecognized.append(item)
    if unrecognized:
        st.warning(f"Unrecognized symptoms: {', '.join(unrecognized)}")
    return diseases_list[model.predict([input_vector])[0]]

def adjust_recommendations(desc, pre, med, die, wrkout, profile):
    adjusted_die = die
    adjusted_wrkout = wrkout
    if 'low_sugar_diet' in profile.get('preferences', []):
        adjusted_die = [d for d in die if 'sugar' not in str(d).lower()] + ['Low-sugar diet recommended']
    if 'light_exercise' in profile.get('preferences', []) and profile.get('location') == 'urban':
        adjusted_wrkout = [w for w in wrkout if 'intense' not in str(w).lower()] + ['Light walking in park']
    return desc, pre, med, adjusted_die, adjusted_wrkout

# User management
users = {
    'user1': {'password': 'pree123', 'role': 'User', 'profile': {'user_id': 1, 'preferences': ['low_sugar_diet', 'light_exercise'], 'location': 'urban', 'time': 'morning'}},
    'admin1': {'password': 'admin123', 'role': 'Admin', 'profile': {}},
    'analyst1': {'password': 'analyst123', 'role': 'Analyst', 'profile': {}}
}

def authenticate_user(username, password):
    if username in users and users[username]['password'] == password:
        return users[username]['role'], users[username]['profile']
    return None, None

# Streamlit app
st.title("Personalized Healthcare Recommendation System")
st.markdown("Enter your credentials and symptoms to get a disease prediction and personalized recommendations.")

# Authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = None
    st.session_state.profile = None

if not st.session_state.authenticated:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            role, profile = authenticate_user(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.profile = profile
                st.success(f"Authenticated as {role}")
            else:
                st.error("Authentication failed.")
else:
    role = st.session_state.role
    profile = st.session_state.profile
    if role in ['User', 'Admin']:
        try:
            with open(MODEL_DIR / 'svc.pkl', 'rb') as f:
                model = pickle.load(f)
            datasets = load_recommendation_data()
            if datasets is None:
                st.stop()
        except FileNotFoundError as e:
            st.error(f"Error: {e}. Ensure model files are in {MODEL_DIR}.")
            st.stop()
        st.subheader("Enter Symptoms")
        symptom_options = list(symptoms_dict.keys())
        selected_symptoms = st.multiselect("Select symptoms (multiple allowed)", symptom_options)
        if st.button("Predict"):
            if selected_symptoms:
                try:
                    predicted_disease = get_predicted_value(selected_symptoms, model)
                    desc, pre, med, die, wrkout = helper(predicted_disease, datasets)
                    desc, pre, med, adjusted_die, adjusted_wrkout = adjust_recommendations(desc, pre, med, die, wrkout, profile)
                    st.subheader("Prediction Results")
                    st.write("**Predicted Disease:**")
                    st.write(predicted_disease)
                    st.write("**Description:**")
                    st.write(desc)
                    st.write("**Precautions:**")
                    for i, p in enumerate(pre, 1):
                        st.write(f"{i}. {p}")
                    st.write("**Medications:**")
                    for i, m in enumerate(med, len(pre)+1):
                        st.write(f"{i}. {m}")
                    st.write("**Adjusted Diet Recommendations:**")
                    for i, d in enumerate(adjusted_die, len(pre)+len(med)+1):
                        st.write(f"{i}. {d}")
                    st.write("**Adjusted Workout Recommendations:**")
                    for i, w in enumerate(adjusted_wrkout, len(pre)+len(med)+len(adjusted_die)+1):
                        st.write(f"{i}. {w}")
                except Exception as e:
                    st.error(f"Error: {e}. Ensure symptoms are valid.")
            else:
                st.warning("Please select at least one symptom.")
    elif role == 'Analyst':
        st.subheader("Analytics Dashboard")
        st.write("**Disease Prediction Accuracy (Simulated):** 100%")
        st.write("More analytics features coming soon!")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.profile = None
        st.experimental_rerun()
