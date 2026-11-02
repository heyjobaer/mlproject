import streamlit as st
import os
from src.ui.uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}
    
    def load_streamlit_ui(self):
        st.set_page_config(page_title=" "+ self.config.get_page_title(), layout='wide')
        st.header(' '+ self.config.get_page_title())
        
        # self.