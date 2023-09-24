from transformers import AutoModelForSequenceClassification
from transformers import RobertaTokenizer
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

import pandas as pd
import plotly.express as px
import numpy as np
from scipy.special import softmax
import warnings

warnings.filterwarnings('ignore')


st.set_page_config(
  page_title='InSocial',
  initial_sidebar_state='expanded',
  layout='wide', 
  page_icon="insocial\IN.png"
)
st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)
hide_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True) 

st.title('In:blue[Social]')
st.markdown('''
            <style>
            h1#insocial{
            padding-top:.3rem;
            padding-bottom: 0;
            font-size: 3.5rem;
            }
            </style>
            ''', unsafe_allow_html=True)
st.caption("Your Sentiment Analyser")
st.divider()

default = "This makes me cry with happiness, I cry because of the pain I’ve felt that Lewis touches on and I’m happy that Lewis describes how I feel which means so many people feel this!!!!"
text = st.text_area('Enter comment/post/tweet', placeholder='Enter text(comment) to analyze', value=default,height=30)
#text = st.text_input('Enter comment/post/tweet', placeholder='Enter text(comment) to analyze', value='It is a nice day')

def word_character_count_frequency(text):
    words = text.split()
    char_count_freq = {}

    for word in words:
        char_count = len(word)
        if char_count in char_count_freq:
            char_count_freq[char_count] += 1
        else:
            char_count_freq[char_count] = 1

    return char_count_freq

freq_dict = word_character_count_frequency(text)
data = {
    'characters':list(freq_dict.keys()),
    'word count':list(freq_dict.values())
}
df = pd.DataFrame(data)

#ML
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

model = AutoModelForSequenceClassification.from_pretrained(r'C:\Users\vedan\OneDrive\Documents\hackthon\sentm')
tokenizer = RobertaTokenizer.from_pretrained(r'C:\Users\vedan\OneDrive\Documents\hackthon\sentm')

model.save_pretrained('sentm')
tokenizer.save_pretrained('sentm')

text_data = preprocess(text)
encoded_input = tokenizer(text_data, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

col1, col2, col3 = st.columns(3)
sentiments = {'Positive':scores[-1], 'Negative':scores[0], 'Neutral':scores[1]}
sent_df = pd.DataFrame(sentiments.items(), columns=['Sentiment', 'Probability'])

sorted_sentiments = dict(sorted(sentiments.items(), reverse=True, 
                           key = lambda x: x[1]))

def sentiment_emoji():
    if list(sorted_sentiments.keys())[0]=='Positive':
        color = '#33b864'
        return f'😊{color}1'
    elif list(sorted_sentiments.keys())[0]=='Negative':
        color = '#bf0a30'
        return f'😓{color}-1'
    else:
        color = '#b9bbb6'
        return f'😌{color}0'

with col1:
    st.subheader('Sentiment probabilities')
    fig = px.pie(sent_df, values='Probability', names='Sentiment', hole=0.5,color='Sentiment',
                 color_discrete_map={'Neutral':'#b9bbb6',
                                 'Positive':'#2093c3',
                                 'Negative':'#2f52a4'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('Overall')
    #emoji card
    
    col2.metric(label='Sentiment emoji', value=sentiment_emoji()[0])

    style_metric_cards(background_color='#2093c3', border_color="#2093c3", border_left_color="#2093c3")


    #overall
    col2.metric(label=f'{list(sorted_sentiments.keys())[0]} probability', 
                value=f'{round(float(list(sorted_sentiments.values())[0]),3)}')
    
    style_metric_cards(background_color='#2093c3', border_color="#2093c3", border_left_color=sentiment_emoji()[1:8])


with col3:
    st.subheader('Word-character count')
    st.bar_chart(df,x='characters', y='word count')
