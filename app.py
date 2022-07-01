import warnings
from explore_page import show_explore_page

warnings.filterwarnings("ignore")

import streamlit as st 
from predict_page import show_predict_page

st.sidebar.title("Explore or Predict")
page = st.sidebar.selectbox("Choose the page you want to use",("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()


    
