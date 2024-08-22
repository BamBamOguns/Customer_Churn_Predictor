import requests
from streamlit_lottie import st_lottie
import streamlit as st
import streamlit.components.v1 as components
import time
import base64
from PIL import Image
from utils.login import invoke_login_widget
from utils.lottie import display_lottie_on_page