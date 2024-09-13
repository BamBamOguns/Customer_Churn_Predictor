import streamlit as st
import seaborn as sns
import plotly.express as px
import pandas as pd
import streamlit_shadcn_ui as ui

# Configure the page
st.set_page_config(
    page_title='Dashboard',
    page_icon = '📈',
    layout = 'wide'
)

st.title("Visualizations")

st.image("assets/Churn_Analysis_FI.png", width = 1000, caption = "Storytelling with visualizations")

# --------- Add custom CSS to adjust the width of the sidebar
st.markdown(""" <style> 
            section[data-testid = "stSidebar"]
            { width: 200px !important;
            }
            </style> """,
            unsafe_allow_html=True,
)

def dashboard_page():
    # Set header for page
    st.title('Dashboard')

    # ------ Set visualization view page
    col1, col2, col3 = st.columns(3)
    with col2:
        options = st.selectbox('Choose viz to display', options=['', 'EDA Dashboard', 'KPIs Dashboard'])

    # ------ Load Dataset from remote location
    @st.cache_data(show_spinner = 'Loading data')
    def load_data():
        df = pd.read_csv('C:\\Users\\HP\\AzubiCA\\Career Accelerator\\LP4\\Customer_Churn_Predictor\\data\\Final_Merged_Data_Cleaned.csv')
        return df

    df = load_data()

    def eda_viz():
        st.subheader('EDA Dashboard')
        column1, column2 = st.columns(2)
        with column1:
            fig = px.histogram(df, x = 'Tenure', title = 'Distribution of Tenure')
            st.plotly_chart(fig)
        with column1:
            fig = px.histogram(df, x = 'MonthlyCharges', title = 'Distribution of MonthlyCharges')
            st.plotly_chart(fig)
        with column1:
            fig = px.histogram(df, x = 'TotalCharges', title = 'Distribution of TotalCharges')
            st.plotly_chart(fig)

        with column2:
            fig = px.bar(df, x = 'Churn', title = 'Churn Distribution')
            st.plotly_chart(fig)
        with column2:
            fig = px.box(df, x = 'Gender', y = 'TotalCharges', title = 'Total Charges Distribution across Gender')
            st.plotly_chart(fig)

    def kpi_viz():
        st.subheader('KPIs Dashboard')
        st.markdown('---')
        cols = st.columns(5)
        st.markdown('---')
        # ------- Grand Total Charges
        with cols[0]:
            grand_tc = df['TotalCharges'].sum()
            ui.metric_card(title = "Grand TotalCharges", content = f"{'{:,.2f}'.format(grand_tc)}", key = "card1")

        # ------- Grand Monthly Charges
        with cols[1]:
            grand_mc = df['MonthlyCharges'].sum()
            ui.metric_card(title = "Grand MonthlyCharges", content = f"{'{:,.2f}'.format(grand_mc)}", key = "card2")

        # ------- Average Customer Tenure
        with cols[2]:
            average_tenure = df['Tenure'].mean()
            ui.metric_card(title = "Average Tenure", content = f"{'{:,.2f}'.format(average_tenure)}", key = "card3")

        # ------- Churned Customers
        with cols[3]:
            churned = len(df.loc[df['Churn'] == 'Yes'])
            ui.metric_card(title = "Churn", content = f"{churned}", key = "card4")

        # ------ Total Customers
        with cols[4]:
            total_customers = df['CustomerID'].count()
            ui.metric_card(title = "Total Customers", content = f"{total_customers}", key = "card5")

    def analytical_ques_viz():
        # ------ Answer Analytical Question 1
        mal_churned_customers = df[(df['Gender'] == 'Male') & (df['Dependents'] == 'Yes') & (df['Churn'] == 'Yes')]['PaymentMethod'].value_counts()

        # Prepare data for visualization
        values = mal_churned_customers.values
        labels = mal_churned_customers.index

        # Create a DataFrame for plotting
        bar_df = pd.DataFrame({'Payment Method': labels, 'Count': values})

        # Create a horizontal bar chart
        fig = px.bar(bar_df, x = 'Count', y = 'Payment Method', orientation = 'h', 
             color = 'Count', color_continuous_scale = 'Blues',
             title = 'Q1. Count of Male Customers with Dependents who Churned by Payment Method')
        
        # Display the plot using Streamlit
        st.plotly_chart(fig)

        # ------ Answer Analytical Question 2
        fem_churned_customers = df[(df['Gender'] =='Female') & (df['Dependents'] == 'Yes') & (df['Churn'] == 'Yes')]['PaymentMethod'].value_counts()

        # Prepare data for visualization
        values = fem_churned_customers.values
        labels = fem_churned_customers.index

        # Create a DataFrame for plotting
        treemap_df = pd.DataFrame({'labels': labels, 'values': values})

        # Create a treemap chart
        fig = px.treemap(treemap_df, path = ['labels'], values = 'values', color = 'values',
                         color_continuous_scale = 'Blues', title = 'Q2. How many female customers with dependents churned given their payment method?')
        st.plotly_chart(fig)

        # ------ Answer Analytical Question 3
        churned = df[df['Churn'] == 'Yes']

        # Create a horizontal bar chart
        fig = px.bar(churned, x = 'MultipleLines', color = 'Gender', barmode = 'group',
                     title = 'Q3. What is the distribution for the customers who churned given their multiple lines status?',
                     labels = {'MultipleLines': 'Multiple Lines', 'Gender': 'Gender'})

        st.plotly_chart(fig)

        # ------ Answer Analytical Question 4
        monthly_charges = df.groupby('Gender')['MonthlyCharges'].sum().reset_index()

        # Create a pie chart for Monthly Charges accumulated by gender
        fig = px.pie(monthly_charges, names = 'Gender', values = 'MonthlyCharges',
                     title = 'Q4. What percentage of MonthlyCharges was accumulated given the customer gender?',
                     color = 'Gender',
                     labels = {'Gender': 'Gender', 'MonthlyCharges': 'Monthly Charges'})

        # Show the figure in Streamlit
        st.plotly_chart(fig)

        # ------ Answer Analytical Question 5
        monthly_charges = df.groupby('Churn')['TotalCharges'].sum().reset_index()

        # Create a pie chart for Total Charges accumulated by churn status
        fig = px.pie(monthly_charges, names = 'Churn', values = 'TotalCharges',
                     title = 'Q5. What percentage of TotalCharges was accumulated given customer churn status?',
                     color = 'Churn',
                     labels = {'Churn': 'Churn', 'TotalCharges': 'Monthly Charges'})

        st.plotly_chart(fig)

    if options == 'EDA Dashboard':
        eda_viz()
    elif options == 'KPIs Dashboard':
        kpi_viz()
        analytical_ques_viz()
    else:
        st.markdown('#### No viz display selected yet')

if __name__ == '__main__':
    dashboard_page()