# PrediCare 

### AI-Powered Early Child Malnutrition Risk Prediction System

PrediCare is a machine learning system that predicts the risk of child malnutrition using NFHS-5 India data.

The system uses a **Random Forest classifier** and **SHAP explainability** to provide not only a risk prediction, but also the major factors contributing to that prediction.

## Key Features

- Child malnutrition risk prediction
- Random Forest machine learning model
- SHAP-based explainability
- Top 3 risk factors for each prediction
- Household Vulnerability Score
- Feeding Quality Score
- Mother-Child Health Index
- Nutrition and government-scheme recommendations
- ASHA Worker mode
- Citizen mode with AI Health Assistant
- Streamlit-based user interface

## Model Performance

- **Algorithm:** Random Forest
- **Accuracy:** 87.25%
- **Recall:** 89.69%
- **Dataset:** NFHS-5 India
- **Children:** 232,920

## Explainable AI

PrediCare uses SHAP to identify the most influential factors behind a model prediction.

Instead of only showing:

`AT RISK`

the system also explains:

`Why is the child at risk?`

and provides the top contributing factors and recommended actions.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SHAP
- Matplotlib
- Streamlit
- Google Gemini API

## Project Structure

```text
PrediCare/
├── app.py
├── PrediCare.ipynb
├── models/
└── outputs/
```

## Run the Project

Clone the repository and install the required Python libraries.

```bash
pip install streamlit pandas numpy scikit-learn shap matplotlib google-generativeai
```

Run the application:

```bash
streamlit run app.py
```

## Disclaimer

PrediCare is a research/educational prototype and is not a medical diagnostic tool. Predictions should not replace professional medical evaluation.
