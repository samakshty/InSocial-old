import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle as pk
import re
import pandas as pd
import pygwalker as pyg
import streamlit.components.v1 as components
from stqdm import stqdm
import plotly.express  as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

hide_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True) 

st.title('Analyze review csv file')
st.markdown('''
      Analyze the dataset of reviews in the form of csv or excel file etc.
      ''')
st.divider()


nltk.download('stopwords')
stop_list = stopwords.words('english')

ps = PorterStemmer()

model = pk.load(open("insocial/revsentm/reviewsentiment.pkl",'rb'))
cv = pk.load(open('insocial/revsentm/vectorizer.pkl', 'rb'))

def clean_text(text):
  text = text.lower()
  text = re.sub(r'<*.?>','', text)
  text = re.sub(r"[^a-zA-Z0-9]+", " ",text)
  text = ' '.join([i for i in text.split() if i not in stop_list])
  data = text.split()
  stemt = list(map(ps.stem,data))
  text = ' '.join(stemt)
  return text


def rev_sent(review):
  mrev = [clean_text(review.strip())]
  feed = cv.transform(mrev)
  pred = model.predict(feed)

  if pred[0]==1:
    return 'pos'
  else:
    return 'neg'
  

file = st.file_uploader("Upload a file", type=(['csv', 'xlsx', 'xls']))

if file is not None:
  if 'csv' in file.type:
    df = pd.read_csv(file, encoding='ISO-8859-1')
  else:
    df = pd.read_excel(file, engine='openpyxl')


  dflist = list(df.columns)
  walker = pyg.walk(df, return_html=True)
  components.html(walker, height=800,scrolling=True)

  features = st.multiselect('Select the column to run sentiment analysis',dflist,default=dflist[0])
  a_button = st.button('Analyze')

  if a_button:
    try:
      selected = df[features]
      
      res = {}
      for i,row in stqdm(selected.iterrows(), total=len(selected)):
        text = row[features[0]]
        res[i] = rev_sent(clean_text(text))

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
      st.error(e)