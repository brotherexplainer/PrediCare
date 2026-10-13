# ============================================
# PrediCare - Child Malnutrition Risk Predictor
# Streamlit Web Application
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="PrediCare",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD ALL ARTIFACTS
# ============================================

@st.cache_resource
def load_models():
    with open('models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    with open('models/recommendations.pkl', 'rb') as f:
        recommendations = pickle.load(f)
    with open('models/shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('models/optimal_threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    return (rf_model, scaler, feature_cols,
            recommendations, explainer, threshold)

rf_model, scaler, feature_cols, recommendations, \
    explainer, optimal_threshold = load_models()

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_and_explain(child_data, top_n=3):
    child_df     = pd.DataFrame([child_data])[feature_cols]
    child_scaled = scaler.transform(child_df)

    proba      = rf_model.predict_proba(child_scaled)[0][1]
    prediction = int(proba >= optimal_threshold)

    shap_raw = explainer.shap_values(child_scaled)

    if isinstance(shap_raw, np.ndarray):
        if len(shap_raw.shape) == 3:
            child_shap = shap_raw[0, :, 1]
        else:
            child_shap = shap_raw[0]
    elif isinstance(shap_raw, list):
        child_shap = np.array(shap_raw[1])[0]
    else:
        child_shap = np.array(shap_raw)[0]

    top_idx = np.argsort(np.abs(child_shap))[::-1][:top_n]

    drivers = []
    for idx in top_idx:
        feat     = feature_cols[idx]
        shap_val = float(child_shap[idx])
        feat_val = float(child_data.get(feat, 0))

        rec = recommendations.get(feat, {})
        if isinstance(rec, dict):
            action_text = rec.get(
                'action', 'Consult nearest PHC'
            )
            feat_name = rec.get('name', feat)
        else:
            action_text = str(rec)
            feat_name   = feat

        drivers.append({
            'feature'    : feat,
            'feat_name'  : feat_name,
            'value'      : round(feat_val, 2),
            'shap_impact': round(shap_val, 4),
            'direction'  : 'Increases risk'
                           if shap_val > 0
                           else 'Lowers risk',
            'action'     : action_text
        })

    if proba >= 0.70:
        confidence = 'HIGH'
    elif proba >= 0.40:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    return {
        'risk_label'    : 'AT RISK'
                          if prediction == 1
                          else 'NOT AT RISK',
        'risk_percent'  : round(float(proba) * 100, 1),
        'confidence'    : confidence,
        'top_drivers'   : drivers,
        'raw_prediction': prediction
    }

# ============================================
# ENGINEERED FEATURES COMPUTATION
# ============================================

def compute_engineered_features(inputs):
    wi  = inputs['wealth_index']
    nt  = inputs['no_toilet']
    uw  = inputs['unsafe_water']
    sf  = inputs['solid_fuel']
    hvs = ((5 - wi) / 4) * 0.30 + nt * 0.25 + \
           uw * 0.25 + sf * 0.20

    bm  = inputs['breastfed_months']
    bs  = inputs['birth_size']
    fqs = (bm / 36) * 0.60 + \
           ((5 - bs) / 4) * 0.40

    bmi = inputs['mother_bmi']
    if 18.5 <= bmi <= 24.9:
        bmi_score = 1.0
    elif bmi > 24.9:
        bmi_score = 0.5
    else:
        bmi_score = 0.0

    edu  = inputs['mother_edu']
    ana  = inputs['mother_anaemia']
    mchi = bmi_score * 0.35 + \
            (1 - ana) * 0.35 + \
            (edu / 3) * 0.30

    return hvs, fqs, mchi


def compute_diet_display_score(
    fqs, meals_per_day, eats_vegetables, eats_protein
):
    """
    DISPLAY-ONLY score for the 'Feeding' health indicator.

    IMPORTANT: this is separate from 'feeding_quality_score'
    (fqs), which is an actual input feature to the trained
    model and must stay computed exactly as the model was
    trained (breastfed_months + birth_size only). Changing
    that formula would feed the model values it never saw
    during training and silently distort predictions.

    This function instead blends fqs with the current-diet
    questions (meals per day, vegetables/fruits, protein)
    purely so the on-screen label reflects the full picture
    the user actually answered. It is never passed to the
    model.
    """
    return (
        (fqs * 0.40)
        + ((meals_per_day / 4) * 0.30)
        + (eats_vegetables * 0.15)
        + (eats_protein * 0.15)
    )

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1565C0;
        text-align: center;
        padding: 1rem 0 0.25rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #FFEBEE;
        border-left: 5px solid #C62828;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #E8F5E9;
        border-left: 5px solid #2E7D32;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .risk-score {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
    }
    .driver-card {
    background-color: #F5F5F5;
    color: #1A1A1A !important;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #C62828;
    }

    .driver-card b {
    color: #1A1A1A !important;
    font-size: 1rem;
    font-weight: 700;
    }
    .action-text {
        background-color: #E3F2FD;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-size: 0.9rem;
        color: #0D47A1;
        margin-top: 0.4rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1565C0;
        margin: 1.5rem 0 0.5rem 0;
        border-bottom: 2px solid #E3F2FD;
        padding-bottom: 0.25rem;
    }
    .stButton > button {
        background-color: #1565C0;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("## PrediCare")
    st.markdown("---")

    mode = st.radio(
        "Select Login Mode",
        ["ASHA Worker (Offline Mode)",
         "Citizen (Online Mode)"],
        index=0
    )

    st.markdown("---")
    st.markdown("**Model Information**")
    st.markdown("- Algorithm: Random Forest")
    st.markdown("- Accuracy: 87.25%")
    st.markdown("- Recall: 89.69%")
    st.markdown("- Dataset: NFHS-5 India")
    st.markdown("- Children: 2,32,920")
    st.markdown("---")
    st.markdown("**Threshold**")
    st.markdown(f"- Value: {optimal_threshold:.4f}")
    st.markdown("- Optimized for 90%+ recall")

# ============================================
# MAIN HEADER
# ============================================

st.markdown(
    '<div class="main-header">PrediCare</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">AI-Powered Child Malnutrition '
    'Risk Prediction System | NFHS-5 India</div>',
    unsafe_allow_html=True
)

# ============================================
# INPUT FORM — CHILD / MOTHER / HOUSEHOLD
# ============================================

st.markdown(
    '<div class="section-title">Child Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Child Details**")
    child_weight = st.number_input(
        "Child Weight (kg x 10, e.g. 95 = 9.5 kg)",
        min_value=15, max_value=300, value=95,
        help="Enter child weight multiplied by 10"
    )
    child_height = st.number_input(
        "Child Height (cm x 10, e.g. 750 = 75 cm)",
        min_value=300, max_value=1300, value=750,
        help="Enter child height multiplied by 10"
    )
    child_age = st.slider(
        "Child Age (months)",
        min_value=0, max_value=59, value=24
    )
    child_sex = st.selectbox(
        "Child Sex",
        options=[0, 1],
        format_func=lambda x: "Male" if x == 0 else "Female"
    )
    birth_size = st.selectbox(
        "Size at Birth",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "Very Small",
            2: "Smaller than Average",
            3: "Average",
            4: "Larger than Average",
            5: "Very Large"
        }[x],
        index=2
    )

with col2:
    st.markdown("**Mother Details**")
    mother_bmi = st.number_input(
        "Mother BMI (e.g. 21.5)",
        min_value=12.0, max_value=50.0,
        value=21.5, step=0.1,
        help="Mother's actual BMI value"
    )
    mother_anaemia = st.selectbox(
        "Mother Anaemia",
        options=[0, 1],
        format_func=lambda x:
            "No Anaemia" if x == 0 else "Has Anaemia"
    )
    mother_edu = st.selectbox(
        "Mother Education Level",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "No Education",
            1: "Primary",
            2: "Secondary",
            3: "Higher Education"
        }[x],
        index=1
    )
    mother_age = st.slider(
        "Mother Age (years)",
        min_value=15, max_value=49, value=25
    )
    age_first_birth = st.slider(
        "Mother Age at First Birth",
        min_value=10, max_value=35, value=20
    )
    teen_mother = st.selectbox(
        "Teen Mother (first birth before 18)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    anc_visits = st.slider(
        "Antenatal Care Visits",
        min_value=0, max_value=8, value=3
    )

with col3:
    st.markdown("**Household Details**")
    wealth_index = st.selectbox(
        "Household Wealth Index",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "Poorest",
            2: "Poor",
            3: "Middle",
            4: "Rich",
            5: "Richest"
        }[x],
        index=1
    )
    rural = st.selectbox(
        "Residence",
        options=[0, 1],
        format_func=lambda x:
            "Urban" if x == 0 else "Rural",
        index=1
    )
    no_toilet = st.selectbox(
        "Toilet Facility",
        options=[0, 1],
        format_func=lambda x:
            "Has Toilet" if x == 0 else "No Toilet"
    )
    unsafe_water = st.selectbox(
        "Drinking Water",
        options=[0, 1],
        format_func=lambda x:
            "Safe Water" if x == 0 else "Unsafe Water"
    )
    solid_fuel = st.selectbox(
        "Cooking Fuel",
        options=[0, 1],
        format_func=lambda x:
            "Clean Fuel" if x == 0 else "Solid Fuel"
    )
    breastfed_months = st.slider(
        "Breastfeeding Duration (months)",
        min_value=0, max_value=36, value=12
    )
    birth_interval = st.slider(
        "Birth Interval (months since last birth)",
        min_value=0, max_value=60, value=24
    )
    under5_siblings = st.slider(
        "Under-5 Siblings in Household",
        min_value=0, max_value=8, value=1
    )

# ============================================
# FOOD AND NUTRITION ASSESSMENT
# (moved above the button so all inputs exist
#  before prediction / scheme logic runs)
# ============================================

st.markdown("---")
st.markdown(
    '<div class="section-title">'
    'Food and Nutrition Assessment'
    '</div>',
    unsafe_allow_html=True
)

food_col1, food_col2 = st.columns(2)

with food_col1:
    meals_per_day = st.selectbox(
        "How many meals does the child eat per day?",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 meal (very poor)",
            2: "2 meals (below recommended)",
            3: "3 meals (recommended)",
            4: "4 or more meals (good)"
        }[x],
        index=1
    )
    eats_vegetables = st.selectbox(
        "Does child eat vegetables or fruits daily?",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )
    eats_protein = st.selectbox(
        "Does child eat protein foods "
        "(eggs, dal, meat, fish)?",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

with food_col2:
    enough_food = st.selectbox(
        "Does family have enough food "
        "throughout the month?",
        options=[0, 1],
        format_func=lambda x:
            "No — food insecure" if x == 0
            else "Yes — food secure"
    )
    has_ration_card = st.selectbox(
        "Does family have a ration card?",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )
    anganwadi_enrolled = st.selectbox(
        "Is child enrolled in "
        "Anganwadi feeding program?",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

# ============================================
# PREDICT BUTTON
# ============================================

st.markdown("---")
predict_clicked = st.button("Predict Malnutrition Risk")

if predict_clicked:

    base_inputs = {
        'child_weight'    : child_weight,
        'child_height'    : child_height,
        'child_age'       : child_age,
        'child_sex'       : child_sex,
        'birth_size'      : birth_size,
        'breastfed_months': breastfed_months,
        'birth_interval'  : birth_interval,
        'under5_siblings' : under5_siblings,
        'mother_bmi'      : mother_bmi,
        'mother_anaemia'  : mother_anaemia,
        'mother_edu'      : mother_edu,
        'mother_age'      : mother_age,
        'age_first_birth' : age_first_birth,
        'teen_mother'     : teen_mother,
        'anc_visits'      : anc_visits,
        'wealth_index'    : wealth_index,
        'rural'           : rural,
        'no_toilet'       : no_toilet,
        'unsafe_water'    : unsafe_water,
        'solid_fuel'      : solid_fuel,
    }

    hvs, fqs, mchi = compute_engineered_features(
        base_inputs
    )

    child_data = base_inputs.copy()
    child_data['household_vulnerability_score'] = hvs
    child_data['feeding_quality_score']         = fqs
    child_data['mother_child_health_index']     = mchi

    with st.spinner("Analyzing child risk..."):
        result = predict_and_explain(child_data, top_n=3)

    diet_display = compute_diet_display_score(
        fqs, meals_per_day, eats_vegetables, eats_protein
    )

    # Persist result across reruns triggered by widgets
    # further down the page (e.g. the chatbot text input)
    st.session_state['predicare_result'] = result
    st.session_state['predicare_hvs'] = hvs
    st.session_state['predicare_fqs'] = fqs
    st.session_state['predicare_mchi'] = mchi
    st.session_state['predicare_diet_display'] = diet_display

# ============================================
# EVERYTHING BELOW ONLY RENDERS ONCE A
# PREDICTION EXISTS IN THIS SESSION
# ============================================

if 'predicare_result' in st.session_state:

    result       = st.session_state['predicare_result']
    hvs          = st.session_state['predicare_hvs']
    fqs          = st.session_state['predicare_fqs']
    mchi         = st.session_state['predicare_mchi']
    diet_display = st.session_state['predicare_diet_display']

    # ============================================
    # RESULT DISPLAY
    # ============================================

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if result['risk_label'] == 'AT RISK':
            st.markdown(f"""
            <div class="risk-high">
                <div class="risk-score"
                     style="color:#C62828;">
                    {result['risk_percent']}%
                </div>
                <div style="text-align:center;
                            font-size:1.4rem;
                            font-weight:700;
                            color:#C62828;">
                    AT RISK
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <div class="risk-score"
                     style="color:#2E7D32;">
                    {result['risk_percent']}%
                </div>
                <div style="text-align:center;
                            font-size:1.4rem;
                            font-weight:700;
                            color:#2E7D32;">
                    NOT AT RISK
                </div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("**Assessment Summary**")
        risk_pct = result['risk_percent']

        if risk_pct >= 70:
            risk_level_text = "High concern — act immediately"
            risk_color      = "#C62828"
        elif risk_pct >= 40:
            risk_level_text = "Moderate concern — monitor closely"
            risk_color      = "#E65100"
        else:
            risk_level_text = "Low concern — continue monitoring"
            risk_color      = "#2E7D32"

        st.markdown(
            f"<span style='color:{risk_color};"
            f"font-weight:600;'>{risk_level_text}</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"- Risk Probability : {result['risk_percent']}%"
        )
        st.markdown(
            f"- Assessment       : {result['risk_label']}"
        )
        st.markdown("- Model Accuracy   : 87.25%")
        st.markdown("- Dataset          : NFHS-5 India")

    with res_col3:
        st.markdown("**Health Indicators**")

        if hvs >= 0.6:
            hvs_text  = "High Risk Environment"
            hvs_color = "#C62828"
        elif hvs >= 0.3:
            hvs_text  = "Moderate Risk Environment"
            hvs_color = "#E65100"
        else:
            hvs_text  = "Safe Environment"
            hvs_color = "#2E7D32"

        if diet_display <= 0.3:
            fqs_text  = "Poor Feeding Quality"
            fqs_color = "#C62828"
        elif diet_display <= 0.6:
            fqs_text  = "Moderate Feeding Quality"
            fqs_color = "#E65100"
        else:
            fqs_text  = "Good Feeding Quality"
            fqs_color = "#2E7D32"

        if mchi <= 0.3:
            mchi_text  = "Mother Needs Support"
            mchi_color = "#C62828"
        elif mchi <= 0.6:
            mchi_text  = "Mother Moderate Health"
            mchi_color = "#E65100"
        else:
            mchi_text  = "Mother Good Health"
            mchi_color = "#2E7D32"

        st.markdown(
            f"- Household : <span style='color:{hvs_color};"
            f"font-weight:600;'>{hvs_text}</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"- Feeding   : <span style='color:{fqs_color};"
            f"font-weight:600;'>{fqs_text}</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"- Mother    : <span style='color:{mchi_color};"
            f"font-weight:600;'>{mchi_text}</span>",
            unsafe_allow_html=True
        )

    # ============================================
    # RISK FACTORS SECTION
    # ============================================

    st.markdown("---")

    if result['risk_label'] == 'AT RISK':
        st.markdown(
            '<div class="section-title">'
            'Top Risk Factors and Recommended Actions'
            '</div>',
            unsafe_allow_html=True
        )

        risk_drivers = [
            d for d in result['top_drivers']
            if d['shap_impact'] > 0
        ]

        if risk_drivers:
            for i, driver in enumerate(risk_drivers, 1):
                st.markdown(f"""
                <div class="driver-card">
                    <b>{i}. {driver['feat_name']}</b>
                    <div class="action-text">
                        Recommended Action: {driver['action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(
                "Child is flagged at risk based on "
                "combined indicators. Please refer to "
                "nearest PHC for full assessment."
            )

    else:
        st.markdown("""
        <div style="background-color:#E8F5E9;
                    border-left:5px solid #2E7D32;
                    padding:1.5rem;
                    border-radius:8px;
                    margin:1rem 0;">
            <b style="color:#2E7D32; font-size:1.1rem;">
                This child does not appear to be at risk
                of malnutrition at this time.
            </b>
            <br><br>
            <span style="color:#1B5E20;">
                Continue regular health check-ups at the
                nearest Anganwadi or PHC. Maintain balanced
                nutrition and breastfeeding as recommended
                by WHO guidelines.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ============================================
    # FOOD SCHEME RECOMMENDATIONS
    # ============================================

    st.markdown("---")
    st.markdown(
        '<div class="section-title">'
        'Food and Nutrition Scheme Recommendations'
        '</div>',
        unsafe_allow_html=True
    )

    food_schemes = []

    # Check food security
    if enough_food == 0 or meals_per_day <= 2:
        food_schemes.append({
            'scheme' : 'PM Garib Kalyan Anna Yojana',
            'reason' : 'Family appears food insecure '
                       'or child eating too few meals',
            'action' : 'Visit nearest ration shop or '
                       'gram panchayat to apply for '
                       'free 5kg grain per person '
                       'per month',
            'urgency': 'HIGH'
        })

    # Check Anganwadi enrollment
    if anganwadi_enrolled == 0:
        food_schemes.append({
            'scheme' : 'ICDS Anganwadi Supplementary '
                       'Nutrition Program',
            'reason' : 'Child not enrolled in '
                       'Anganwadi feeding program',
            'action' : 'Enroll child at nearest '
                       'Anganwadi center for free '
                       'daily supplementary nutrition '
                       'for children under 6 years',
            'urgency': 'HIGH'
        })

    # Check ration card
    if has_ration_card == 0:
        food_schemes.append({
            'scheme' : 'National Food Security Act '
                       '(NFSA) Ration Card',
            'reason' : 'Family does not have a '
                       'ration card',
            'action' : 'Apply for ration card at '
                       'nearest tehsil or gram '
                       'panchayat office to access '
                       'subsidized food grains',
            'urgency': 'MEDIUM'
        })

    # Check protein intake
    if eats_protein == 0:
        food_schemes.append({
            'scheme' : 'POSHAN Abhiyaan — National '
                       'Nutrition Mission',
            'reason' : 'Child lacks protein in diet',
            'action' : 'Contact nearest Anganwadi '
                       'worker for nutrition '
                       'counselling on protein-rich '
                       'foods (eggs, dal, fish) '
                       'available under POSHAN scheme',
            'urgency': 'MEDIUM'
        })

    # Check vegetables
    if eats_vegetables == 0:
        food_schemes.append({
            'scheme' : 'POSHAN Abhiyaan — Dietary '
                       'Diversity Program',
            'reason' : 'Child not eating vegetables '
                       'or fruits daily',
            'action' : 'ASHA worker to counsel '
                       'family on importance of '
                       'dietary diversity — include '
                       'green leafy vegetables '
                       'and seasonal fruits daily',
            'urgency': 'LOW'
        })

    if food_schemes:
        for scheme in food_schemes:
            if scheme['urgency'] == 'HIGH':
                border_color  = "#C62828"
                urgency_color = "#C62828"
            elif scheme['urgency'] == 'MEDIUM':
                border_color  = "#E65100"
                urgency_color = "#E65100"
            else:
                border_color  = "#1565C0"
                urgency_color = "#1565C0"

            st.markdown(f"""
            <div style="background-color:#FFFFFF;
                        border-left:4px solid {border_color};
                        border-radius:8px;
                        padding:1rem;
                        margin:0.5rem 0;
                        color:#1A1A1A;">
                <b style="color:{border_color};">
                    {scheme['scheme']}
                </b>
                <span style="float:right;
                             color:{urgency_color};
                             font-size:0.8rem;
                             font-weight:600;">
                    {scheme['urgency']} PRIORITY
                </span>
                <br>
                <span style="color:#546E7A;
                             font-size:0.9rem;">
                    Why: {scheme['reason']}
                </span>
                <div style="background-color:#E3F2FD;
                            border-radius:6px;
                            padding:0.5rem 0.75rem;
                            margin-top:0.4rem;
                            font-size:0.9rem;
                            color:#0D47A1;">
                    Action: {scheme['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success(
            "Family appears food secure. "
            "Continue current nutrition practices "
            "and maintain regular Anganwadi visits."
        )

    # ============================================
    # ASHA WORKER — REFERRAL ALERT
    # ============================================

    if "ASHA" in mode and result['risk_label'] == 'AT RISK':
        st.markdown("---")
        st.error(
            "HIGH PRIORITY: This child requires immediate "
            "follow-up. Please refer to the nearest PHC "
            "or Anganwadi center for further assessment."
        )

    # ============================================
    # CITIZEN MODE — AI CHATBOT
    # ============================================

    if "Citizen" in mode:
        st.markdown("---")
        st.markdown(
            '<div class="section-title">'
            'AI Health Assistant'
            '</div>',
            unsafe_allow_html=True
        )
        st.info(
            "Ask any question about your child's "
            "malnutrition risk, nutrition, or next steps."
        )

        user_question = st.text_input(
            "Your Question",
            placeholder=(
                "e.g. What food should I give my child?"
            )
        )

        if user_question:
            try:
                import google.generativeai as genai
                genai.configure(
                    api_key=st.secrets["GEMINI_API_KEY"]
                )
                # NOTE: 'gemini-pro' is deprecated in current
                # google-generativeai releases — use a current
                # model name instead.
                model_ai = genai.GenerativeModel(
                    'gemini-1.5-flash'
                )
                context = (
                    f"A child has been assessed by "
                    f"PrediCare, an AI malnutrition "
                    f"risk prediction system. "
                    f"Risk score: {result['risk_percent']}%. "
                    f"Status: {result['risk_label']}. "
                    f"Top risk factors: "
                    f"{', '.join([d['feat_name'] for d in result['top_drivers']])}. "
                    f"Answer this parent's question in "
                    f"simple, clear language suitable "
                    f"for a rural Indian parent. "
                    f"Question: {user_question}"
                )
                with st.spinner("Getting response..."):
                    response = model_ai.generate_content(
                        context
                    )
                    st.markdown(
                        f"**AI Response:** {response.text}"
                    )
            except Exception as e:
                st.warning(
                    "AI chatbot requires a valid Gemini API "
                    "key. Add GEMINI_API_KEY to "
                    ".streamlit/secrets.toml "
                    f"(error: {e})"
                )

# ============================================
# VISUALIZATIONS
# ============================================


# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center;"
    "color:#90A4AE;font-size:0.85rem;'>"
    "PrediCare | NFHS-5 India | "
    "Random Forest 87.25% Accuracy | "
    "Department of Electronics and Computer "
    "Science Engineering"
    "</div>",
    unsafe_allow_html=True
)
