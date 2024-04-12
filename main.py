import streamlit as st
import pandas as pd
from utils.functions import *

with st.sidebar:
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

st.title('💬 FAQ Chatbot')
file_name = 'banking.csv'
create_vector_db(file_name)
st.caption('🚀 Your friendly banking assistant.')
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you today?'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input():
    st.session_state.messages.append({'role':'user', 'content': prompt})
    st.chat_message('user').write(prompt)
    # st.write(st.session_state.messages)
    chain = get_qa_chain()
    response = chain(prompt)
    st.chat_message('llm').write(response['result'])
