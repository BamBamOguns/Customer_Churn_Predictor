import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title = "Welcome Page",
    page_icon = "assets/app_icon.svg",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

st.title("Customer Churn Prediction App")

# Display the customer icon/picture

st.image("assets/c_churn_1.png", width = 1000)

# Welcome Message

st.markdown("""
            
    ## Welcome to the Customer Churn Prediction App!
    
    This app is designed to help businesses predict customer churn, explore insights, and make data-driven decisions to enhance customer retention. By leveraging data visualization and machine learning, this tool empowers you to identify at-risk customers and take proactive measures.
    
    ### How to Navigate the App
    
    - **Home Page:** Start here to learn about the app's features and purpose. The home page provides a guide on how to use each section effectively, helping you get the most out of your experience.
    
    - **Data Page:** Access the Churn Dataset to explore key features and understand the data used for churn analysis. Here, you can view, filter, and analyze the data, providing a solid foundation for understanding the factors influencing customer churn.
    
    - **Dashboard Page:** Visualize important patterns and trends within the data using interactive charts and graphs. This section helps you spot correlations and insights that might not be immediately obvious from raw data.
    
    - **History Page:** Review past predictions, analysis logs, and other recorded activities. This page allows you to track the history of your analysis efforts, providing context for ongoing decision-making.
    
    - **Prediction Page:** Generate churn predictions based on customer data inputs. This section utilizes the trained model to forecast which customers are likely to churn, allowing you to take preventative actions.
    
    - **Sign Up Page:** Register for personalized access and updates. Signing up gives you the ability to save your predictions, customize your experience, and receive alerts based on churn analysis.
    
    Explore each page using the sidebar navigation. Dive into the data, gain insights, and make informed decisions to improve your business's customer retention strategy!
            
            """)