import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
        page_title = "Data Navigator",
        page_icon = "📊", 
        layout = "wide")
    
st.title("Data Navigator")

st.image("assets/c_churn_3.jpeg", width = 1000, caption = "Unlock your potentials using Data")
    
st.markdown("""
    ### Explore and Analyze Your Data

    Welcome to the Data Navigator, your comprehensive tool for exploring, analyzing, and preparing your data. This page is designed to help you uncover valuable insights hidden within your datasets. With the Data Navigator, you can:

    - **Upload Your Data**: Seamlessly import your data in various formats, including CSV, Excel, and JSON, allowing you to work with your own datasets effortlessly.
    - **Explore Sample Datasets**: Dive right in with pre-loaded sample datasets to get familiar with the platform and understand how to use the available features.
    - **Understand Data Structure**: Access detailed descriptions of each column, including data types and content expectations, to ensure your data aligns correctly with the analysis requirements.
    - **Filter and Refine**: Use interactive filters and sorting options to focus on specific subsets of your data, helping you identify patterns and trends more effectively.
    - **Review Data Summary**: View a comprehensive summary of your dataset, including key statistics like count, mean, median, and more, to get an overview of your data at a glance.

    With the Data Navigator, you have all the tools you need to explore, understand, and prepare your data for further analysis and visualization.
    """)

# --------- Add custom CSS to adjust the width of the sidebar
st.markdown( """ <style> 
            section[data-testid="stSidebar"]
            { width: 200px !important;
            }
            </style> """,
            unsafe_allow_html=True,
)

def data_page():
    # Set header for dataset view
    st.title('Dataset View')

    # Create selection option
    column1, column2 = st.columns(2)
    with column2:
        option = st.selectbox('Choose columns to be viewed',
                              ('All Columns', 'Numeric Columns', 'Categorical Columns'))

    # ---- Load dataset function
    @st.cache_data(show_spinner = 'Loading data')
    def load_data():
        # Update the file path to a relative path within the project directory
        df = pd.read_csv('C:/Users/HP/AzubiCA/Career Accelerator/LP4/Customer_Churn_Predictor/data/Final_Merged_Data_Cleaned.csv')
        return df

    # Load data and display the first 100 rows
    df = load_data().head(100)

    # Display data based on user selection
    if option == 'Numeric Columns':
        st.subheader('Numeric Columns')
        st.write(df.select_dtypes(include = 'number'))

    elif option == 'Categorical Columns':
        st.subheader('Categorical Columns')
        st.write(df.select_dtypes(include = 'object'))

    else:
        st.subheader('Complete Dataset')
        st.write(df)

    # ----- Add column descriptions of the dataset
    with st.expander('**Click to view column description**'):
        st.markdown('''
            :gray[**The following describes the columns present in the data.**]

        **Gender** -- Whether the customer is a male or a female

        **SeniorCitizen** -- Whether a customer is a senior citizen or not

        **Partner** -- Whether the customer has a partner or not (Yes, No)

        **Dependents** -- Whether the customer has dependents or not (Yes, No)

        **Tenure** -- Number of months the customer has stayed with the company

        **Phone Service** -- Whether the customer has a phone service or not (Yes, No)

        **MultipleLines** -- Whether the customer has multiple lines or not

        **InternetService** -- Customer's internet service provider (DSL, Fiber Optic, No)

        **OnlineSecurity** -- Whether the customer has online security or not (Yes, No, No Internet)

        **OnlineBackup** -- Whether the customer has online backup or not (Yes, No, No Internet)

        **DeviceProtection** -- Whether the customer has device protection or not (Yes, No, No internet service)

        **TechSupport** -- Whether the customer has tech support or not (Yes, No, No internet)

        **StreamingTV** -- Whether the customer has streaming TV or not (Yes, No, No internet service)

        **StreamingMovies** -- Whether the customer has streaming movies or not (Yes, No, No Internet service)

        **Contract** -- The contract term of the customer (Month-to-Month, One year, Two year)

        **PaperlessBilling** -- Whether the customer has paperless billing or not (Yes, No)

        **Payment Method** -- The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))

        **MonthlyCharges** -- The amount charged to the customer monthly

        **TotalCharges** -- The total amount charged to the customer

        **Churn** -- Whether the customer churned or not (Yes or No)
        ''')

if __name__ == '__main__':
    data_page()

