import streamlit as st
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import joblib
import os

# Configure the page
st.set_page_config(
    page_title = 'Predictions',
    page_icon = '🔮',
    layout = 'wide'
)

st.title("Churn Predictor")

st.image("assets/c_churn_4.jpeg", width = 1000, caption = "Are you Churning?")

# --------- Add custom CSS to adjust the width of the sidebar
st.markdown(""" 
    <style> 
        section[data-testid="stSidebar"] { width: 200px !important; }
    </style> """, unsafe_allow_html = True)

st.title("Predict Customer Churn!")

column1, column2 = st.columns([.6, .4])
with column1:
    model_option = st.selectbox('Choose which model to use for prediction', options = ['Gradient Boosting', 'Random Forest'])

# Define file paths
local_model1_path = 'C:\\Users\\HP\\AzubiCA\\Career Accelerator\\LP4\\Customer_Churn_Predictor\\models\\gradient_boosting_model.pkl'
local_model2_path = 'C:\\Users\\HP\\AzubiCA\\Career Accelerator\\LP4\Customer_Churn_Predictor\\models\\random_forest_model.pkl'

# -------- Function to load the model from local files
@st.cache_resource(show_spinner = "Loading model")
def gb_pipeline():
    model = joblib.load(local_model1_path)
    return model

@st.cache_resource(show_spinner = "Loading model")
def rf_pipeline():
    model = joblib.load(local_model2_path)
    return model

# --------- Create a function for model selection
def select_model():
    # Choose the model based on user selection
    if model_option == 'Gradient Boosting':
        model = gb_pipeline()
    else:  # If 'Random Forest' is selected
        model = rf_pipeline()
    
    # Only return the selected model
    return model

# Custom function to deal with cleaning the total charges column
class TotalCharges_cleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
        
    def transform(self, X):
        # Replace empty string with NA
        X['TotalCharges'].replace(' ', np.nan, inplace = True)
        # Convert the values in the TotalCharges column to a float
        X['TotalCharges'] = X['TotalCharges'].astype(float)
        return X
        
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass
        
    def get_feature_names_out(self, input_features = None):
        return input_features

# Create a class to deal with dropping Customer ID from the dataset
class columnDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
        
    def transform(self, X):
        return X.drop('CustomerID', axis = 1)
        
    def get_feature_names_out(self, input_features = None):
        if input_features is None:
            return None
        return [feature for feature in input_features if feature != 'CustomerID']

# Initialize prediction in session state
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'prediction_proba' not in st.session_state:
    st.session_state['prediction_proba'] = None

# ------- Create a function to make prediction
def make_prediction(model):
    # Extract input data from session state
    CustomerID = st.session_state['Customer_id']
    Gender = st.session_state['Gender']
    SeniorCitizen = st.session_state['Senior_Citizen']
    Partner = st.session_state['Partners']
    Dependents = st.session_state['Dependents']
    Tenure = st.session_state['Tenure']
    PhoneService = st.session_state['Phone_Service']
    MultipleLines = st.session_state['Multiple_Lines']
    InternetService = st.session_state['Internet_Service']
    OnlineSecurity = st.session_state['Online_Security']
    OnlineBackup = st.session_state['Online_Backup']
    DeviceProtection = st.session_state['Device_Protection']
    TechSupport = st.session_state['Tech_Support']
    StreamingTV = st.session_state['Streaming_TV']
    StreamingMovies = st.session_state['Streaming_Movies']
    Contract = st.session_state['Contract']
    PaperlessBilling = st.session_state['Paperless_Billing']
    PaymentMethod = st.session_state['Payment_Method']
    MonthlyCharges = st.session_state['Monthly_Charges']
    TotalCharges = st.session_state['Total_Charges']

     # Define column names  
    columns = ['CustomerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

    # Create a DataFrame with input values  
    values = [[CustomerID, Gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService,
            MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
            PaymentMethod, MonthlyCharges, TotalCharges]]
        
    data = pd.DataFrame(values, columns = columns)

    # Get the value for prediction
    prediction = model.predict(data)
    st.session_state['prediction'] = prediction

    # Get the value for prediction probability
    prediction_proba = model.predict_proba(data)
    st.session_state['prediction_proba'] = prediction_proba

    data['Churn'] = prediction
    data['Model'] = model_option

    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    data.to_csv('./data/history.csv', mode = 'a', header = not os.path.exists('./data/history.csv'), index = False)

    return prediction, prediction_proba

# ------- Prediction page creation
def input_features():
    with st.form('features'):
       # Call the select_model function which now returns only the model
        model_pipeline = select_model()
        col1, col2 = st.columns(2)

        # ------ Collect customer information
        with col1:
            st.subheader('Demographics')
            Customer_ID = st.text_input('Customer ID', value = "", placeholder = 'eg. 1234-ABCDE')
            Gender = st.radio('Gender', options = ['Male', 'Female'], horizontal = True)
            Partners = st.radio('Partners', options = ['Yes', 'No'], horizontal = True)
            Dependents = st.radio('Dependents', options = ['Yes', 'No'], horizontal = True)
            Senior_Citizen = st.radio("Senior Citizen ('Yes-1, No-0')", options = [1, 0], horizontal = True)
            
        # ------ Collect customer account information
        with col1:
            st.subheader('Customer Account Info.')
            Tenure = st.number_input('Tenure', min_value = 0, max_value = 70)
            Contract = st.selectbox('Contract', options = ['Month-to-month', 'One year', 'Two year'])
            Payment_Method = st.selectbox('Payment Method',
                                          options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            Paperless_Billing = st.radio('Paperless Billing', ['Yes', 'No'], horizontal = True)
            Monthly_Charges = st.number_input('Monthly Charges', placeholder = 'Enter amount...')
            Total_Charges = st.number_input('Total Charges', placeholder = 'Enter amount...')
            
        # ------ Collect customer subscription information
        with col2:
            st.subheader('Subscriptions')
            Phone_Service = st.radio('Phone Service', ['Yes', 'No'], horizontal = True)
            Multiple_Lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No internet service'])
            Internet_Service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            Online_Security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
            Online_Backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
            Device_Protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
            Tech_Support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
            Streaming_TV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
            Streaming_Movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])

        # Add the submit button
        submitted = st.form_submit_button('Predict')
        if submitted:
            # Save input data into session state
            st.session_state['Customer_ID'] = Customer_ID
            st.session_state['Gender'] = Gender
            st.session_state['Senior_Citizen'] = Senior_Citizen
            st.session_state['Partners'] = Partners
            st.session_state['Dependents'] = Dependents
            st.session_state['Tenure'] = Tenure
            st.session_state['Phone_Service'] = Phone_Service
            st.session_state['Multiple_Lines'] = Multiple_Lines
            st.session_state['Internet_Service'] = Internet_Service
            st.session_state['Online_Security'] = Online_Security
            st.session_state['Online_Backup'] = Online_Backup
            st.session_state['Device_Protection'] = Device_Protection
            st.session_state['Tech_Support'] = Tech_Support
            st.session_state['Streaming_TV'] = Streaming_TV
            st.session_state['Streaming_Movies'] = Streaming_Movies
            st.session_state['Contract'] = Contract
            st.session_state['Paperless_Billing'] = Paperless_Billing
            st.session_state['Payment_Method'] = Payment_Method
            st.session_state['Monthly_Charges'] = Monthly_Charges
            st.session_state['Total_Charges'] = Total_Charges
            
            # Make the prediction
            make_prediction(model_pipeline)

st.divider()
    
# Show the prediction result
st.subheader('Prediction Results')

# Display the churn prediction
if st.session_state['prediction'] is not None:
        if st.session_state['prediction'][0] == 1:
            st.success(f'This customer is predicted to Churn with probability {np.round(st.session_state["prediction_proba"][0][1] * 100, 2)}%')
        else:
            st.info(f'This customer is predicted to Stay with probability {np.round(st.session_state["prediction_proba"][0][0] * 100, 2)}%')
else:
        st.write("Click 'Predict' to generate the result!")

input_features()