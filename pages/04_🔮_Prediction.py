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
        section[data-testid = "stSidebar"] { width: 200px !important; }
    </style> """, unsafe_allow_html = True)

st.title("Predict Customer Churn!")

column1, column2 = st.columns([.6, .4])
with column1:
    model_option = st.selectbox('Choose which model to use for prediction', options = ['Gradient Boosting', 'Random Forest'])

# Define file paths
local_model1_path = './models/gradient_boosting_model.pkl'
local_model2_path = './models/random_forest_model.pkl'
local_encoder_path = './models/label_encoder.joblib'

# -------- Function to load the model from local files
@st.cache_resource(show_spinner = "Loading model")
def gb_pipeline():
    model = joblib.load(local_model1_path)
    return model

@st.cache_resource(show_spinner = "Loading model")
def rf_pipeline():
    model = joblib.load(local_model2_path)
    return model

# --------- Function to load encoder from local files
def load_encoder():
    encoder = joblib.load(local_encoder_path)
    return encoder

# --------- Create a function for model selection
def select_model():
    # Choose the model based on user selection
    if model_option == 'Gradient Boosting':
        model = gb_pipeline()
    else:  # Ensure 'Random Forest' is handled correctly
        model = rf_pipeline()
    encoder = load_encoder()
    return model, encoder

# Custom function to deal with cleaning the total charges column
class TotalCharges_cleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
        
    def transform(self, X):
        # Replace empty string with NaN
        X['TotalCharges'].replace(' ', np.nan, inplace = True)
        
        # Fill NaN values with a default or average value, or handle as desired
        X['TotalCharges'].fillna(0, inplace = True)  # Replace with 0 or an appropriate value
        
        # Convert the values in the TotalCharges column to a float
        try:
            X['TotalCharges'] = X['TotalCharges'].astype(float)
        except ValueError as e:
            st.error(f"Error converting 'TotalCharges': {e}")
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

# Initialize all other session state keys to avoid KeyError
session_keys = [
    'CustomerID', 'Gender', 'SeniorCitizen', 'Partners', 'Dependents', 'Tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'MonthlyCharges', 'TotalCharges'
]

# Set default values for each key if not already initialized
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = ""

# ------- Create a function to make prediction
def make_prediction(model, encoder):
    # Extract input data from session state
    CustomerID = st.session_state.get('CustomerID', '')
    Gender = st.session_state.get('Gender', '')
    SeniorCitizen = st.session_state.get('SeniorCitizen', '')
    Partner = st.session_state.get('Partner', '')
    Dependents = st.session_state.get('Dependents', '')
    Tenure = st.session_state.get('Tenure', '')
    PhoneService = st.session_state.get('PhoneService', '')
    MultipleLines = st.session_state.get('MultipleLines', '')
    InternetService = st.session_state.get('InternetService', '')
    OnlineSecurity = st.session_state.get('OnlineSecurity', '')
    OnlineBackup = st.session_state.get('OnlineBackup', '')
    DeviceProtection = st.session_state.get('DeviceProtection', '')
    TechSupport = st.session_state.get('TechSupport', '')
    StreamingTV = st.session_state.get('StreamingTV', '')
    StreamingMovies = st.session_state.get('StreamingMovies', '')
    Contract = st.session_state.get('Contract', '')
    PaperlessBilling = st.session_state.get('PaperlessBilling', '')
    PaymentMethod = st.session_state.get('PaymentMethod', '')
    MonthlyCharges = st.session_state.get('MonthlyCharges', '')
    TotalCharges = st.session_state.get('TotalCharges', '')

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

    # Handle empty strings and missing values for numeric columns
    numeric_columns = ['Tenure', 'MonthlyCharges', 'TotalCharges']

    # Replace empty strings with NaN and convert numeric columns to float
    data[numeric_columns] = data[numeric_columns].replace('', np.nan)  # Replace empty strings with NaN
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors = 'coerce')  # Convert to numeric, force errors to NaN        # Convert columns to float
    
    # Fill NaN values with a default value or an appropriate strategy (like 0 or mean)
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace = True)
    data['MonthlyCharges'].fillna(data['MonthlyCharges'].mean(), inplace = True)
    data['Tenure'].fillna(0, inplace = True)  # Fill tenure with 0 if empty

    try:
        # Get the value for prediction
        prediction = model.predict(data)
        prediction = encoder.inverse_transform(prediction)
        st.session_state['prediction'] = prediction

        # Get the value for prediction probability
        prediction_proba = model.predict_proba(data)
        st.session_state['prediction_proba'] = prediction_proba

        # Append the prediction and model used to the data
        data['Churn'] = prediction
        data['Model'] = model_option

        # Save the prediction history
        if not os.path.exists('./data'):
            os.makedirs('./data')
        
        data.to_csv('./data/history.csv', mode = 'a', header = not os.path.exists('./data/history.csv'), index = False)

        return prediction, prediction_proba
    
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        st.write("Please check the input values and try again.")

# ------- Prediction page creation
def input_features():
    with st.form('features'):
       # Call the select_model function which now returns only the model
        model_pipeline, encoder = select_model()
        col1, col2 = st.columns(2)

        # ------ Collect customer information
        with col1:
            st.subheader('Demographics')
            CustomerID = st.text_input('Customer ID', value = "", placeholder = 'eg. 1234-ABCDE')
            Gender = st.radio('Gender', options = ['Male', 'Female'], horizontal = True)
            Partners = st.radio('Partners', options = ['Yes', 'No'], horizontal = True)
            Dependents = st.radio('Dependents', options = ['Yes', 'No'], horizontal = True)
            SeniorCitizen = st.radio("Senior Citizen ('Yes-1, No-0')", options = [1, 0], horizontal = True)
            
        # ------ Collect customer account information
        with col1:
            st.subheader('Customer Account Info.')
            Tenure = st.number_input('Tenure', min_value = 0, max_value = 70)
            Contract = st.selectbox('Contract', options = ['Month-to-month', 'One year', 'Two year'])
            PaymentMethod = st.selectbox('Payment Method',
                                          options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            PaperlessBilling = st.radio('Paperless Billing', ['Yes', 'No'], horizontal = True)
            MonthlyCharges = st.number_input('Monthly Charges', placeholder = 'Enter amount...')
            TotalCharges = st.number_input('Total Charges', placeholder = 'Enter amount...')
            
        # ------ Collect customer subscription information
        with col2:
            st.subheader('Subscriptions')
            PhoneService = st.radio('Phone Service', ['Yes', 'No'], horizontal = True)
            MultipleLines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No internet service'])
            InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            OnlineSecurity = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
            OnlineBackup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
            DeviceProtection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
            TechSupport = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
            StreamingTV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
            StreamingMovies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])

        # Add the submit button
        submitted = st.form_submit_button('Predict')
        if submitted:
            # Save input data into session state
            st.session_state['Customer_ID'] = CustomerID
            st.session_state['Gender'] = Gender
            st.session_state['Senior_Citizen'] = SeniorCitizen
            st.session_state['Partners'] = Partners
            st.session_state['Dependents'] = Dependents
            st.session_state['Tenure'] = Tenure
            st.session_state['Phone_Service'] = PhoneService
            st.session_state['Multiple_Lines'] = MultipleLines
            st.session_state['Internet_Service'] = InternetService
            st.session_state['Online_Security'] = OnlineSecurity
            st.session_state['Online_Backup'] = OnlineBackup
            st.session_state['Device_Protection'] = DeviceProtection
            st.session_state['Tech_Support'] = TechSupport
            st.session_state['Streaming_TV'] = StreamingTV
            st.session_state['Streaming_Movies'] = StreamingMovies
            st.session_state['Contract'] = Contract
            st.session_state['Paperless_Billing'] = PaperlessBilling
            st.session_state['Payment_Method'] = PaymentMethod
            st.session_state['Monthly_Charges'] = MonthlyCharges
            st.session_state['Total_Charges'] = TotalCharges
            
            # Make the prediction
            make_prediction(model_pipeline, encoder)

    return True

if __name__ == '__main__':
    input_features()
    
    prediction = st.session_state['prediction']
    probability = st.session_state['prediction_proba']    

    if prediction is None:
        cols = st.columns([3, 4, 3])
        with cols[1]:
            st.markdown('#### Predictions will show here ⤵️')
        cols = st.columns([.25, .5, .25])
        with cols[1]:
            st.markdown('##### No predictions made yet. Make a prediction.')
    else:
        if prediction == "Yes":
            cols = st.columns([.1, .8, .1])
            with cols[1]:
                st.markdown(f'### The customer will churn with a {round(probability[0][1], 2)} probability.')
            cols = st.columns([.3, .4, .3])
            with cols[1]:
                st.success('Churn status predicted successfully🎉')
        else:
            cols = st.columns([.1, .8, .1])
            with cols[1]:
                st.markdown(f'### The customer will not churn with a {round(probability[0][0], 2)} probability.')
            cols = st.columns([.3, .4, .3])
            with cols[1]:
                st.success('Churn status predicted successfully🎉')

#st.divider()
    
# Show the prediction result
#st.subheader('Prediction Results')

# Display the churn prediction
#if st.session_state['prediction'] is not None:
       # if st.session_state['prediction'][0] == 1:
        #    st.success(f'This customer is predicted to Churn with probability {np.round(st.session_state["prediction_proba"][0][1] * 100, 2)}%')
        #else:
        #    st.info(f'This customer is predicted to Stay with probability {np.round(st.session_state["prediction_proba"][0][0] * 100, 2)}%')
#else:
        #st.write("Click 'Predict' to generate the result!")

#input_features()