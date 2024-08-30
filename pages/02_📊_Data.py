import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
        page_title = "Data Navigator",
        page_icon = "assets/app_icon.svg", 
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