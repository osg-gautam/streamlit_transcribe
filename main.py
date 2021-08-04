import streamlit as st
from transcribe import *
import time
import pandas as pd
import numpy as np
import json
import plotly.express as px

import sys
import time
import requests
import numpy as np
import json
from datetime import datetime
import text2emotion as te
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
analyzer = SentimentIntensityAnalyzer()
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import en_core_web_sm

import spacy
nlp = spacy.load('en_core_web_sm')
####################################################################
sdb = pd.read_csv('tx_speaker_db.csv')
wdb = pd.read_csv('tx_word_db.csv')
####################################################################
st.image('osglogo.jpg', use_column_width=False)
st.sidebar.image('osglogo.jpg', use_column_width=False)
st.sidebar.title('Coversation Analysis')
option = st.sidebar.selectbox('Please select',('Welcome','Upload new conversation', 
'Load previous conversation'))

st.subheader(option)

if option == 'Welcome':
    st.subheader('Welcome to Conversation analysis dashboard.')
    st.subheader('It enables you to do following tasks:')
    st.markdown('* Upload audio files in any format, transcribe and save conversation in database.')
    st.markdown('* Open and analyze previously transcripted files')


if option == 'Upload new conversation':
    fileObject = st.file_uploader(label = "Please upload audio")
    st.audio(fileObject, format='audio/wav')
    if fileObject:
        fname = fileObject.name
        if fname in wdb.fname.values:
            st.subheader('File exist in database')
            st.subheader('File details:')
            dfw = wdb[wdb.fname==fname]
            st.subheader('Summary of uploaded file')
            st.write('Conversation ID: ',fname)
            st.write('Count of speakers: ',dfw.speaker.nunique())
            st.write('Date created     : ',dfw.iloc[0]['datetime'])
            st.write('Duration in mins  : ',round(dfw.iloc[0]['duration']/60))
            
        else:
            token, t_id = upload_file(fileObject)
            result = {}
            #polling
            sleep_duration = 1
            percent_complete = 0
            progress_bar = st.progress(percent_complete)
            st.text("Currently in queue")
            while result.get("status") != "processing":
                percent_complete += sleep_duration
                time.sleep(sleep_duration)
                progress_bar.progress(percent_complete/10)
                result = get_text(token,t_id)

            sleep_duration = 0.01

            for percent in range(percent_complete,101):
                time.sleep(sleep_duration)
                progress_bar.progress(percent)

            with st.spinner("Processing....."):
                while result.get("status") != 'completed':
                    result = get_text(token,t_id)

            st.balloons()

            ## process word db
            audindex = pd.json_normalize(result['words'])
            audindex['fname'] = fname
            audindex['duration'] = result['audio_duration']
            audindex['datetime'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            #audindex['comments'] = st.text_input("add comments", 'add comments if any')
            audindex.to_csv('tx_word_db.csv',mode='a',header=False)

            ## process speaker db

            speakers = list(audindex.speaker) #Change df to your dataframe name
            previous_speaker = 'A'
            l = len(speakers)
            i = 1
            speaker_seq_list = list()
            for index, new_speaker in enumerate(speakers):
                if index > 0:
                    previous_speaker = speakers[index - 1]
                if new_speaker != previous_speaker:
                    i+=1
                speaker_seq_list.append(i)
                #print(str(previous_speaker)+"  "+str(new_speaker)+"  "+str(i))
            audindex['seq'] = speaker_seq_list
            df=pd.DataFrame(audindex.groupby(['fname','speaker','seq']).agg(utter = ('text',' '.join),stime=('start','min'),etime=('end','max')))
            df.reset_index(inplace=True)
            df.sort_values(by=['stime'],inplace=True)

            df['stime'] = df.stime//1000
            df['etime'] = df.etime//1000
            df['seq'] = df.seq-1
            ## add word count
            df['wcount'] = df['utter'].apply(lambda x: len(x.split()))
            df.reset_index(inplace=True)
            df.drop('index',axis=1,inplace=True)

            ## emotion 
            df['emotion'] = df.utter.apply(lambda x:te.get_emotion(x))
            df['emotion'] = df.emotion.apply(lambda x:max(x, key=x.get))

            ## sentiment
            df['senti'] = [analyzer.polarity_scores(x)['compound'] for x in df['utter']]

            def sentclass(x):
                if x >= 0.05 :
                    return "Positive"
                elif x <= - 0.05 :
                    return "Negative"
                else :
                    return "Neutral"
                
            df['senti'] = [sentclass(x) for x in df['senti']]

            ## key phrases
            hilites=pd.json_normalize(result['auto_highlights_result']['results'])
            hilites = hilites.text.unique()

            df['key_phrase'] = 'none'
            for x in hilites:
                df.loc[(df.utter.str.contains(x)),'key_phrase']=x

            df.to_csv('tx_speaker_db.csv',mode='a',header=False,index=False)

            st.write('conversation added to database')

            ###########################
            #sdb = pd.read_csv('tx_speaker_db.csv')

            ## summary of the file uploaded
            wdb = pd.read_csv('tx_word_db.csv')
            dfw = wdb[wdb.fname==fname]
            st.subheader('Summary of uploaded file')
            st.write('Conversation ID: ',fname)
            st.write('Count of speakers: ',dfw.speaker.nunique())
            #st.markdown('Speakers         :', dfw.speaker.unique())
            st.write('Date created     : ',dfw.iloc[0]['datetime'])
            st.write('Duration in mins  : ',round(dfw.iloc[0]['duration']/60))

            
if option == 'Load previous conversation':
    wdb = pd.read_csv('tx_word_db.csv')
    sdb = pd.read_csv('tx_speaker_db.csv')
    fx = st.selectbox('Please select conversation',(sdb.fname.unique()))
    st.subheader('Summary')
    
    ## summary of the file uploaded
    dfw = wdb[wdb.fname==fx]
 
    st.write('Conversation ID  : ',fx)
    st.write('Count of speakers: ',dfw.speaker.nunique())
    #st.markdown('Speakers         :', dfw.speaker.unique())
    st.write('Date created     : ',dfw.iloc[0]['datetime'])
    st.write('Duration in mins  : ',round(dfw.iloc[0]['duration']/60))
    
    sdb = pd.read_csv('tx_speaker_db.csv')
    df = sdb[sdb.fname==fx]
    dfsum = pd.DataFrame(df.groupby(['speaker']).agg(utter = ('utter',' '.join)))
    dfsum['share'] = dfsum['utter'].apply(lambda x: len(x.split()))
    dfsum['share'] = round(dfsum.share/dfsum.share.sum()*100)
   
    dfsum['summary'] = ''
    for ind in dfsum.index:
        dfsum['summary'][ind]=summarize(dfsum['utter'][ind], ratio = 0.05)#ratio = 0.05 #word_count = 200

    dfsum.reset_index(inplace=True)
    st.subheader('Speakerwise Summarization of Conversation')
    for i in dfsum.index:
                st.markdown(dfsum.iloc[i]['speaker'])
                st.write(dfsum.iloc[i]['summary'])

    #######################################    ## visuals

    c1, c2 = st.beta_columns(2)

    with c1:
        st.header("Key Phrases")
        dfk = pd.DataFrame(df.key_phrase.value_counts())
        dfk.reset_index(inplace=True)
        dfk.columns = ['key_phrases','count']
        dfk = dfk[dfk.key_phrases!='none']
        st.table(dfk,)

        #st.header('Emotions')


    with c2:
        st.header("Speakers Talktime Share")
        fig = px.pie(dfsum, values='share', names='speaker',color='speaker')
        st.plotly_chart(fig)

    st.header("Conversation Timeline")
    
    df1 = sdb[sdb.fname==fx]
    fig1 = px.bar(df1, x="seq", y="wcount", color="speaker", title="Timeline", hover_data=["utter",'emotion','senti'],)
    fig1.update_layout(xaxis={'categoryorder':'category ascending'})
    st.plotly_chart(fig1,width=3000,height=1500,   use_container_width=True )

    ## select utterence based on seq

   
    ## display converstaion

    st.header("Key Phrase Context")

    kx = st.selectbox('Please select Key Phrase',(dfk.key_phrases.unique()))

    dfz = df1[df1.key_phrase==kx]
    dfz.reset_index(inplace=True)

    for i in dfz.index:
        st.subheader(dfz.iloc[i]['speaker'] +' at sequence ' + str(dfz.iloc[i]['seq']))
        st.write(dfz.iloc[i]['utter'])

    ## question answers

    #st.header("Question Answers")

    #dfq = df1[df1.utter.str.contains("?", regex=False)]
    #st.table(dfq)





    