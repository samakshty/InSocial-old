import streamlit as st
from streamlit_extras.let_it_rain import rain


st.title('About')
hide_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True) 

st.markdown('''
            This project was created for **Bennett University Internal Hackathon 2023**.

            **Team Members:** 
            
            [Aditi Singh](https://www.linkedin.com/in/aditi-singh-43937227b) (Team leader)\n
            [Sukant Aryan](https://www.linkedin.com/in/sukantaryan)\n
            [Akshit Gulati](https://www.linkedin.com/in/akshit-gulati-5a70b128b)\n
            [Samaksh Tyagi](https://www.linkedin.com/in/samakshtyagi)\n
            [Vaishnavi](https://www.linkedin.com/in/vaishnavi-a2151628a)\n
            [Vedansh Vats](https://www.linkedin.com/in/vedanshvats2307)
            ''')

rain(
  emoji='✨',
  font_size=54,
  falling_speed=10,
  animation_length=50
)