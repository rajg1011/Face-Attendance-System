import streamlit as st

st.set_page_config(page_title='Attendance System',layout='wide')
st.header("Attendance Project")

with st.spinner("loading Models and connecting to Redis DB..."):
    import face_rec 

st.success("Model Loaded Successfully")
st.success("Data Successfully retrieved from database")

