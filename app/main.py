import streamlit as st

# Hide the default sidebar
st.set_page_config(initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.error("请通过正确的URL访问对应的应用：\n- /annotation") 