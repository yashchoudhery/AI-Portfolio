import streamlit as st
import requests


st.set_page_config(page_title="Portfolio",layout="centered")
st.title("Magical Portfolio")
st.write("Dear recruiters, Ask Any thing About Yash Choudhery!")

query=st.text_area("Ask Any thing about me(how much he qualified? what are his skills? etc..)",height=150,placeholder="Enter your query")

API_URL="http://127.0.0.1:9999/chat"
if st.button("Search"):
    if query.strip():
        payload={ "model_name": "llama-3.3-70b-versatile",
                  "model_provider": "Groq",
                  "prompt":"Act as AI Assistant",
                  "messages": [query],
                  "allow_search": False }
        response = requests.post(API_URL, json=payload)
        if response.status_code != 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Response")
                st.markdown(response_data)
        else:
            response_data = response.json()
            st.subheader("Response")
            st.markdown(response_data)