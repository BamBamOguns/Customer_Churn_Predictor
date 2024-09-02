import streamlit as st
import pandas as pd
import os
import hashlib

# Configure the page
st.set_page_config(
    page_title = 'Sign Up',
    page_icon = '✍🏾',
    layout = 'centered'
)

st.title("Sign Up with US!")

st.image("assets/signup.png", width = 1000, caption = "Subscribe with US!")

# Set file path for user data
user_data_path = './data/users.csv'

# Ensure the data directory exists
if not os.path.exists('./data'):
    os.makedirs('./data')

# Function to hash passwords
def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

# Function to load user data
def load_user_data():
    """Loads user data from the CSV file."""
    if os.path.exists(user_data_path):
        return pd.read_csv(user_data_path)
    else:
        # Create an empty DataFrame if the file doesn't exist
        return pd.DataFrame(columns = ['username', 'password_hash', 'email'])

# Function to save user data
def save_user_data(df):
    """Saves user data to the CSV file."""
    df.to_csv(user_data_path, index = False)

# Function to check if the username already exists
def username_exists(username, df):
    """Checks if the username already exists in the user data."""
    return username in df['username'].values

# Function to validate email format
def validate_email(email):
    """Validates the email format using a basic check."""
    if "@" in email and "." in email:
        return True
    return False

# Function to handle the signup process
def signup(username, password, email):
    """Handles the signup process by validating and saving user data."""
    user_data = load_user_data()

    # Check if the username already exists
    if username_exists(username, user_data):
        st.error('Username already exists. Please choose a different username.')
        return False

    # Validate email format
    if not validate_email(email):
        st.error('Invalid email address. Please enter a valid email.')
        return False

    # Hash the password
    password_hash = hash_password(password)

    # Save the new user to the DataFrame
    new_user = pd.DataFrame({
        'username': [username],
        'password_hash': [password_hash],
        'email': [email]
    })
    user_data = pd.concat([user_data, new_user], ignore_index = True)

    # Save the updated user data
    save_user_data(user_data)
    st.success('Account created successfully! You can now log in.')
    return True

# Create a signup form
st.title('Sign Up for Churn Predictor App')
st.write("Create a new account to start using the Churn Predictor App.")

with st.form('signup_form'):
    username = st.text_input('Username', placeholder = 'Enter your username')
    password = st.text_input('Password', type = 'password', placeholder = 'Enter your password')
    confirm_password = st.text_input('Confirm Password', type = 'password', placeholder = 'Re-enter your password')
    email = st.text_input('Email', placeholder = 'Enter your email address')

    # Create a submit button
    submitted = st.form_submit_button('Sign Up')

    # Check if the form is submitted
    if submitted:
        # Validate that all fields are filled
        if not username or not password or not confirm_password or not email:
            st.error('Please fill in all fields.')
        elif password != confirm_password:
            st.error('Passwords do not match. Please try again.')
        else:
            # Proceed with the signup process
            signup(username, password, email)
