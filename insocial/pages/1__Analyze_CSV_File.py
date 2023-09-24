import streamlit as st
import pandas as pd
import re
from transformers import AutoModelForSequenceClassification
from transformers import RobertaTokenizer
import numpy as np
from scipy.special import softmax
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from stqdm import stqdm
import plotly.express  as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pygwalker as pyg
import streamlit.components.v1 as components


nltk.download('vader_lexicon')

#settings
st.set_page_config(layout='wide')
hide_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True) 

st.set_option('deprecation.showPyplotGlobalUse', False)


model = AutoModelForSequenceClassification.from_pretrained(r'C:\Users\vedan\OneDrive\Documents\hackthon\sentm')
tokenizer = RobertaTokenizer.from_pretrained(r'C:\Users\vedan\OneDrive\Documents\hackthon\sentm')


st.title('Analyze csv file')
st.markdown('''
            Upload a csv file containing the comment/tweet/post data and relax. **Let it analyze for you!**
            *(You must specify the name of the columns to be analyzed)*
            ''')


st.divider()

def clean_text(text):
  text = text.lower()
  text = re.sub(r'@\w+','', text)
  text = re.sub(r'http\S+', '', text)
  return text

def return_pred(text):
  pred_text = clean_text(text)
  encoded_input = tokenizer(pred_text, return_tensors='pt')
  output = model(**encoded_input)
  scores = output[0][0].detach().numpy()
  
  mapping = {'negative':scores[0], 'neutral':scores[1], 'positive':scores[-1]}
  sort_map = sorted(mapping.items(), reverse=True, key = lambda x : x[1])

  return sort_map[0][0]

siq = SentimentIntensityAnalyzer()
def labelvader(text):
  a = siq.polarity_scores(text)
  a.popitem()
  scores  = sorted(a.items(), reverse=True, key = lambda x:x[1])
  return scores[0][0]



file = st.file_uploader("Upload a file", type=(['csv', 'xlsx', 'xls']))
if file is not None:
  if 'csv' in file.type:
    df = pd.read_csv(file, encoding='ISO-8859-1')
  else:
    df = pd.read_excel(file, engine='openpyxl')

  st.subheader('Loaded data')
  dflist = list(df.columns)

  #walker component
  walker = pyg.walk(df, return_html=True)
  components.html(walker, height=800,scrolling=True)
  #--

  features = st.multiselect('Select the column to run sentiment analysis',dflist,default=dflist[0])
  a_button = st.button('Analyze')

  if a_button:
    try:
      selected = df[features]
      
      res = {}
      #with st.spinner('In progress...'):
      for i,row in stqdm(selected.iterrows(), total=len(selected)):
        text = row[features[0]]
        res[i] = labelvader(clean_text(text))

      result = pd.DataFrame(list(res.items()))
      result.columns=['in', 'sentiment']

      result = result.drop('in', axis=1)

      merged = selected.merge(result, left_index=True, right_index=True, how='inner')

      st.write(merged)

      csv = merged.to_csv(index=False).encode('utf-8')
      st.download_button('Download labelled data as csv', data=csv, file_name='export.csv', 
                        help='Click here to download the data as csv file', mime='text/csv')

      col1, col2 = st.columns(2)
      #plots
      with col1:
        st.subheader('Sentiment frequency')
        xl = merged['sentiment'].value_counts().values
        yl = merged['sentiment'].value_counts().index

        barfig =  px.bar(df, x =xl , y=yl,template='seaborn')
        
        st.plotly_chart(barfig, use_container_width=True)

      #wordcloud
      with col2:
        st.subheader('Word Cloud')
        merged[features[0]] = merged[features[0]].apply(clean_text)
        wordcloud2 = WordCloud().generate(' '.join(merged[features[0]]))
        plt.imshow(wordcloud2, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        st.pyplot()
    except Exception as e:
      st.error('An error occurred')
      
















# col1, col2, col3 = st.columns(3)

# with col1:
#   ind1 = st.number_input('Content column index', min_value=0)

# with col2:
#   ind2 = st.number_input('Name column index', min_value=0)

# with col3:
#   ind3 = st.number_input('Date column index', min_value=0)







  

#if a_button:
  # if file is not None:
  #   # ar = [ind1, ind2, ind3]
  #   # ar = [x for x in ar if x != 0]
  #   # ar = list(map(lambda x: x-1, ar))
  #   # selected =df.iloc[:, ar]

  #   # res = {}
  #   # for i,row in stqdm(selected.iterrows(), total=len(df)):
  #   #   text = row['content']
  #   #   res[i] = labelvader(text)
    
  #   # result = pd.DataFrame(list(res.items()))
  #   # result.columns=['in', 'sentiment']

  #   # result = result.drop('in', axis=1)

  #   # merged = selected.merge(result, left_index=True, right_index=True, how='inner')

  #   # st.write(merged)
  
  # else:
  #   st.error('Add data file to analyze')


  











