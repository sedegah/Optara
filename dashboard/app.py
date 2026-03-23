import requests
import streamlit as st

st.set_page_config(page_title="Optara Dashboard", layout="wide")
st.title("Optara Facial Recognition Dashboard")

api_base = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000/api")
refresh = st.sidebar.button("Refresh logs")

if refresh or True:
    try:
        response = requests.get(f"{api_base}/logs/", timeout=5)
        response.raise_for_status()
        logs = response.json()
        st.metric("Recent Events", len(logs))
        st.dataframe(logs, use_container_width=True)
    except Exception as exc:
        st.warning(f"Unable to load logs from API: {exc}")
