import streamlit as st

st.set_page_config(
    page_title="AI Model Showcase", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.page_link('pages/model.py', label='Model')
st.sidebar.page_link('pages/about.py', label='About')

st.switch_page("pages/model.py")