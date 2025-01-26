import streamlit as st
from lecgen.demo import ScriptViewer

# Hide the default sidebar
st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize and run the script generator
viewer = ScriptViewer()
viewer.render() 