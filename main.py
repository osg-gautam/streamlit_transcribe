import streamlit as st
from transcribe import *
import time
import pandas as pd
import numpy as np
import json
import plotly.express as px
import os
import base64
cwd = os.getcwd()


st.header("Transcription Dashboard")

fileObject = st.file_uploader(label = "Please upload audio" )
st.audio(fileObject, format='audio/wav')

if fileObject:
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

    st.header("Transcribed Text")
    st.write(result['text'])
    df = pd.json_normalize(result['words'])
    # audindex['size'] = audindex.end - audindex.start

    # c1, c2 = st.beta_columns(2)
    #
    # with c1:
    # 	st.markdown("Highlights")
    # 	dfx = pd.json_normalize(result['auto_highlights_result']['results'])
    # 	dfx = dfx[['count','text']]
    # 	st.write(dfx)
    #
    # with c2:
    # 	st.markdown("Speakers Talktime Share")
    # 	temp = pd.DataFrame(audindex.speaker.value_counts())
    # 	temp.reset_index(inplace=True)
    # 	fig = px.pie(temp, values='speaker', names='index', title='Share')
    # 	st.plotly_chart(fig)


    def get_table_download_link(df):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        # base = os.path.basename('Welcome.wav')
        filename = os.path.splitext(fileObject.name)[0]
        # filename = fileObject.name
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download csv file</a>'
        return href


    st.markdown(get_table_download_link(df), unsafe_allow_html=True)
