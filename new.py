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

import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from string import punctuation
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# st.set_page_config(page_title="Conversation Analysis",layout='wide')

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

def data_preprocessing(para):
    
    stop_words = list(set(stopwords.words('english')))
    stop_words.extend(['say','alright','would','could',"okay","maybe"])
    
    def clean(para):    
        punct = set(string.punctuation)
        punc_free = " ".join([i for i in para.lower().split() if i not in punct])
        clean_doc = re.sub('[^A-Za-z]+', ' ',punc_free)
#         print(clean_doc)
        return clean_doc
    
    clean_para = clean(para)
#     nlp = spacy.load('en_core_web_sm')
#     allowed_postags=['NOUN']  
    allowed_postags = ['NOUN','VERB','ADV','ADJ']
    
    def lemmatization(paragraph, allowed_postags):
        texts_lemmatized = []
        texts_stopfree = []
        doc = nlp(paragraph)
#         texts_lemmatized.extend([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        texts_lemmatized = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
#         print(texts_lemmatized)
        texts_stopfree = [text for text in texts_lemmatized if text not in stop_words]
        
        texts_stopfree = " ".join(texts_stopfree)
        return texts_stopfree
     
    data_lemmatized = lemmatization(clean_para, allowed_postags)
        
    return data_lemmatized


def tf_idf(doc_list):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(doc_list)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    dense_list = dense.tolist()
    df = pd.DataFrame(dense_list, columns=feature_names)
    return df

def word_cloud(df):
    cloud = WordCloud(width = 1000, height = 700,min_font_size = 10, background_color="white", max_words=80,stopwords = stopwords).generate_from_frequencies(df.T.sum(axis=1))                       
    plt.figure(figsize = (5, 5), facecolor = None)
    fig, axes = plt.subplots()
    axes.imshow(cloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    # plt.show()
    st.pyplot(fig)



def phrase_match(phrase):
    i=0
    indices = []
    for sent in sdb.utter:
        if(phrase in sent.lower()):
            if(i==0):
                indices.extend([i,i+1])
            elif(i==len(sdb.utter)-1):
                indices.extend([i,i-1])
            else:
                indices.extend([i-1,i,i+1])

        i=i+1
    indices = list(set(indices))
    matched_phrases_df = pd.DataFrame(sdb.utter.loc[indices])
    # matched_phrases_df = matched_phrases_df.style.where(lambda val:phrase in val, 'color: green', subset=['utter'])
    return list(matched_phrases_df.utter)

# def pie_chart(labels_list,sizes_list):
#     chart = plt.pie(sizes_list, labels=labels_list)
#     plt.axis('equal')
#     plt.figure(figsize = (10, 10))
#     fig, axes = plt.subplots()
#     axes.imshow(chart)
#     # plt.show()
#     st.pyplot(fig)









st.image('osglogo.jpg', use_column_width=False)
st.sidebar.image('osglogo.jpg', use_column_width=False)
st.sidebar.title('Coversation Analysis')
option = st.sidebar.selectbox('Please select', ('Welcome', 'Upload new conversation',
                                                'Load previous conversation'))

st.subheader(option)

if option == 'Welcome':
    st.subheader('Welcome to Conversation analysis dashboard.')
    st.subheader('It enables you to do following tasks:')
    st.markdown('* Upload audio files in any format, transcribe and save conversation in database.')
    st.markdown('* Open and analyze previously transcripted files')

if option == 'Upload new conversation':
    fileObject = st.file_uploader(label="Please upload audio")
    st.audio(fileObject, format='audio/wav')
    if fileObject:
        fname = fileObject.name
        if fname in wdb.fname.values:
            st.subheader('File exist in database')
            st.subheader('File details:')
            dfw = wdb[wdb.fname == fname]
            st.subheader('Summary of uploaded file')
            st.write('Conversation ID: ', fname)
            st.write('Count of speakers: ', dfw.speaker.nunique())
            st.write('Date created     : ', dfw.iloc[0]['datetime'])
            st.write('Duration in mins  : ', round(dfw.iloc[0]['duration'] / 60))

        else:
            token, t_id = upload_file(fileObject)
            result = {}
            # polling
            sleep_duration = 1
            percent_complete = 0
            progress_bar = st.progress(percent_complete)
            st.text("Currently in queue")
            while result.get("status") != "processing":
                percent_complete += sleep_duration
                time.sleep(sleep_duration)
                progress_bar.progress(percent_complete / 10)
                result = get_text(token, t_id)

            sleep_duration = 0.01

            for percent in range(percent_complete, 101):
                time.sleep(sleep_duration)
                progress_bar.progress(percent)

            with st.spinner("Processing....."):
                while result.get("status") != 'completed':
                    result = get_text(token, t_id)

            st.balloons()

            ## process word db
            audindex = pd.json_normalize(result['words'])
            audindex['fname'] = fname
            audindex['duration'] = result['audio_duration']
            audindex['datetime'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            # audindex['comments'] = st.text_input("add comments", 'add comments if any')
            audindex.to_csv('tx_word_db.csv', mode='a', header=False)

            ## process speaker db

            speakers = list(audindex.speaker)  # Change df to your dataframe name
            previous_speaker = 'A'
            l = len(speakers)
            i = 1
            speaker_seq_list = list()
            for index, new_speaker in enumerate(speakers):
                if index > 0:
                    previous_speaker = speakers[index - 1]
                if new_speaker != previous_speaker:
                    i += 1
                speaker_seq_list.append(i)
                # print(str(previous_speaker)+"  "+str(new_speaker)+"  "+str(i))
            audindex['seq'] = speaker_seq_list
            df = pd.DataFrame(
                audindex.groupby(['fname', 'speaker', 'seq']).agg(utter=('text', ' '.join), stime=('start', 'min'),
                                                                  etime=('end', 'max')))
            df.reset_index(inplace=True)
            df.sort_values(by=['stime'], inplace=True)

            df['stime'] = df.stime // 1000
            df['etime'] = df.etime // 1000
            df['seq'] = df.seq - 1
            ## add word count
            df['wcount'] = df['utter'].apply(lambda x: len(x.split()))
            df.reset_index(inplace=True)
            df.drop('index', axis=1, inplace=True)

            ## emotion
            df['emotion'] = df.utter.apply(lambda x: te.get_emotion(x))
            df['emotion'] = df.emotion.apply(lambda x: max(x, key=x.get))

            ## sentiment
            df['senti'] = [analyzer.polarity_scores(x)['compound'] for x in df['utter']]


            def sentclass(x):
                if x >= 0.05:
                    return "Positive"
                elif x <= - 0.05:
                    return "Negative"
                else:
                    return "Neutral"


            df['senti'] = [sentclass(x) for x in df['senti']]

            ## key phrases
            hilites = pd.json_normalize(result['auto_highlights_result']['results'])
            hilites = hilites.text.unique()

            df['key_phrase'] = 'none'
            for x in hilites:
                df.loc[(df.utter.str.contains(x)), 'key_phrase'] = x

            df.to_csv('tx_speaker_db.csv', mode='a', header=False, index=False)

            st.write('conversation added to database')

            ###########################
            sdb = pd.read_csv('tx_speaker_db.csv')

            ## summary of the file uploaded
            wdb = pd.read_csv('tx_word_db.csv')
            dfw = wdb[wdb.fname == fname]
            st.subheader('Summary of uploaded file')
            st.write('Conversation ID: ', fname)
            st.write('Count of speakers: ', dfw.speaker.nunique())
            # st.markdown('Speakers         :', dfw.speaker.unique())
            st.write('Date created     : ', dfw.iloc[0]['datetime'])
            st.write('Duration in mins  : ', round(dfw.iloc[0]['duration'] / 60))

if option == 'Load previous conversation':
    wdb = pd.read_csv('tx_word_db.csv')
    sdb = pd.read_csv('tx_speaker_db.csv')
    fx = st.selectbox('Please select conversation', (sdb.fname.unique()))
    st.subheader('Summary')

    ## summary of the file uploaded
    dfw = wdb[wdb.fname == fx]

    st.write('Conversation ID  : ', fx)
    st.write('Count of speakers: ', dfw.speaker.nunique())
    # st.markdown('Speakers         :', dfw.speaker.unique())
    st.write('Date created     : ', dfw.iloc[0]['datetime'])
    st.write('Duration in mins  : ', round(dfw.iloc[0]['duration'] / 60))

    sdb = pd.read_csv('tx_speaker_db.csv')
    df = sdb[sdb.fname == fx]
    dfsum = pd.DataFrame(df.groupby(['speaker']).agg(utter=('utter', ' '.join)))
    dfsum['share'] = dfsum['utter'].apply(lambda x: len(x.split()))
    dfsum['share'] = round(dfsum.share / dfsum.share.sum() * 100)

    dfsum['summary'] = ''
    for ind in dfsum.index:
        dfsum['summary'][ind] = summarize(dfsum['utter'][ind], ratio=0.05)  # ratio = 0.05 #word_count = 200

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
        dfk.columns = ['key_phrases', 'count']
        dfk = dfk[dfk.key_phrases != 'none']
        st.table(dfk, )

        # st.header('Emotions')

    with c2:
        st.header("Speakers Talktime Share")
        fig = px.pie(dfsum, values='share', names='speaker', color='speaker')
        st.plotly_chart(fig)

    st.header("Conversation Timeline")

    df1 = sdb[sdb.fname == fx]
    fig1 = px.bar(df1, x="seq", y="wcount", color="speaker", title="Timeline",
                  hover_data=["utter", 'emotion', 'senti'], )
    fig1.update_layout(xaxis={'categoryorder': 'category ascending'})
    st.plotly_chart(fig1, width=3000, height=1500, use_container_width=True)

    ## select utterence based on seq

    ## display converstaion
######################################################################################################################
    st.header("Speaker wise  Word Cloud")
    c1, c2 = st.beta_columns(2)
    with c1:
        st.header("Speaker A")
        docA = [sent for sent in sdb[sdb['speaker']=='A'].utter]
        docA = [data_preprocessing(para) for para in docA]
        tfidf_dfA = tf_idf(docA)
        word_cloud(tfidf_dfA)
        

    with c2:
        st.header("Speaker B")
        docB = [sent for sent in sdb[sdb['speaker']=='B'].utter]
        docB = [data_preprocessing(para) for para in docB]
        tfidf_dfB = tf_idf(docB)
        word_cloud(tfidf_dfB)

############################################################################

    st.header("Key Phrase Context")

    kp = st.text_input("Please enter Key Phrase","phobia")
    phrase_match_df = phrase_match(str(kp))
    for phrase in phrase_match_df:
        st.write(phrase)

    # kx = st.selectbox('Please select Key Phrase',(dfk.key_phrases.unique()))

    # dfz = df1[df1.key_phrase==kx]
    # dfz.reset_index(inplace=True)

    # for i in dfz.index:
    #     st.subheader(dfz.iloc[i]['speaker'] +' at sequence ' + str(dfz.iloc[i]['seq']))
    #     st.write(dfz.iloc[i]['utter'])



##################################################################################################


    st.header("Speaker wise Sentiments")

    # beta_cont_senti = st.beta_container()
    c1, c2 = st.beta_columns(2)
    with c1:
        st.header("Speaker A")
        sentiment_dict = dict(sdb[sdb['speaker']=='A'].senti.value_counts())        
        sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=["labels","values"])
        fig = px.pie(sentiment_df, values='values', names='labels',hole=.3)
        # fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig)

        # for label, size in sentiment_dict.items():
        #     sentiment_labels.append(label)
        #     sentiment_size.append(size)
        # pie_chart(sentiment_labels,sentiment_size)

    with c2:
        st.header("Speaker B")
        sentiment_dict = dict(sdb[sdb['speaker']=='B'].senti.value_counts())        
        sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=["labels","values"])
        fig = px.pie(sentiment_df, values='values', names='labels',hole=.3)
        # fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig)

        # sentiment_labels,sentiment_size = [],[]
        # sentiment_dict = dict(sdb[sdb['speaker']=='B'].senti.value_counts())
        # for label, size in sentiment_dict.items():
        #     sentiment_labels.append(label)
        #     sentiment_size.append(size)
        # pie_chart(sentiment_labels,sentiment_size)

##################################################################################


    st.header("Speaker wise Emotions")
    # beta_cont_emo = st.beta_container()
    c1, c2 = st.beta_columns(2)
    with c1:
        st.header("Speaker A")
        emotion_dict = dict(sdb[sdb['speaker']=='A'].emotion.value_counts())        
        emotion_df = pd.DataFrame(emotion_dict.items(),columns=["labels","values"])
        fig = px.pie(emotion_df, values='values', names='labels')
        st.plotly_chart(fig, use_column_width=False)

    with c2:
        st.header("Speaker B")
        emotion_dict = dict(sdb[sdb['speaker']=='B'].emotion.value_counts())        
        emotion_df = pd.DataFrame(emotion_dict.items(),columns=["labels","values"])
        fig = px.pie(emotion_df, values='values', names='labels')
        st.plotly_chart(fig, use_column_width=False)