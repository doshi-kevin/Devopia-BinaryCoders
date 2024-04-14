import streamlit as st
import requests

url = "https://03d3-34-16-190-37.ngrok-free.app/chat"  # Replace with your API endpoint

st.title("Chatbot")
st.write("Enter the subjects and get a tailored advice.")

subjects = st.text_input(
    "Subjects (comma-separated):",
    "Math 100,Science 100,Social Science 43,English 56,Hindi 34",
)

if st.button("Get Advice"):
    parameters = {"text": subjects}
    response = requests.get(url, params=parameters)

    if response.status_code == 200:
        data = response.json()
        st.json(data)
    else:
        st.write("Error:", response.status_code)
