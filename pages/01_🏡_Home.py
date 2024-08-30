# pages/01_🏡_Home.py
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title = "Customer Churn Prediction - Home",
    page_icon = "assets/app_icon.svg",
    layout = "wide")

st.title("About the Customer Churn Prediction App")

st.image("assets/c_churn_2.png", width = 1000, caption = "Why leave us?")

# Home Page Content

st.markdown("""
    ## The Game Changer App!

    The Customer Churn Prediction App is a powerful tool designed to help businesses identify customers who are likely to stop using their services. By leveraging machine learning models and data visualization techniques, this app provides valuable insights that can help companies take proactive measures to improve customer retention and reduce churn rates.

    ### Purpose of the App

    In today’s competitive market, retaining customers is more cost-effective than acquiring new ones. This app aims to empower businesses by providing predictive analytics that highlight at-risk customers. By understanding the factors that contribute to churn, companies can make data-driven decisions to enhance their strategies and improve customer satisfaction.

    ### How to Navigate the App

    The app is structured into several pages, each serving a specific purpose. Below is an overview of each page and how to use them:

    1. **Home Page**  
       The starting point of the app, this page provides an overview of the app's purpose, features, and navigation guide. It's designed to help users understand the app's benefits and how to make the most out of it.

    2. **Data Page**  
       Here, you can explore the Churn Dataset used for the predictions. The data page connects to a Microsoft SQL Server and allows you to view, filter, and analyze the data. Understanding the data structure and the key features helps you gain insights into the factors influencing churn.

    3. **Dashboard Page**  
       The dashboard provides interactive data visualizations using Plotly or Seaborn. You can explore customer behavior patterns, identify key metrics, and visualize trends that contribute to customer churn. The graphical representations make it easier to spot correlations and make informed decisions.

    4. **History Page**  
       This page provides a record of past predictions and analysis logs. Reviewing past predictions helps track the accuracy of the model and provides a historical context for ongoing decision-making. It’s a valuable resource for analyzing trends over time.

    5. **Prediction Page**  
       On the prediction page, you can generate churn predictions by inputting customer data. The page utilizes the trained machine learning model to forecast which customers are likely to churn, allowing you to identify at-risk individuals and take action.

    6. **Sign-Up Page**  
       The sign-up page enables you to register for personalized access and updates. By signing up, you can save your predictions, customize your experience, and receive notifications based on the latest churn analysis.

    ### Getting the Most Out of the App

    - **Explore the Data:** Start by familiarizing yourself with the data on the Data Page to understand what drives customer churn.
    - **Visualize Trends:** Use the Dashboard Page to spot trends and correlations that can guide your business decisions.
    - **Review Past Predictions:** Leverage the History Page to see how past data compares to current trends, refining your approach over time.
    - **Make Predictions:** Utilize the Prediction Page to identify at-risk customers and make data-driven interventions.
    - **Stay Connected:** Sign up to save your work and receive updates on the latest churn predictions.

    By navigating through these pages, you can gain a comprehensive understanding of your customer base, identify key drivers of churn, and implement targeted retention strategies. Start exploring and take control of your customer churn challenges today!
    """)