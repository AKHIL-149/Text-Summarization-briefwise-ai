import streamlit as st
import requests

st.title("Text Summarization")

text = st.text_area("Enter your text:", height=200)

if st.button("Summarize"):
    response = requests.post(
        "http://localhost:5100/api/summarize",
        json={"text": text}
    )
    result = response.json()
    st.json(result)
