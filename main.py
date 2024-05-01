import datetime
import os
import uuid
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from utils.functions import create_vector_db, \
    get_qa_chain, create_db_table, insert_into_table, read_db

st.set_page_config(page_title='FAQ Chatbot',
                   page_icon=':lips:', layout='centered')

# FILEPATH = 'https://raw.githubusercontent.com/atonui/pds/main/banking.csv'
FILEPATH = 'banking.csv'

with st.sidebar:
    selected = option_menu(None, ['Chat', 'History', 'About', 'Data'],
                           icons=['chat-dots', 'clock-history', 'info-circle', 'database'],
                           menu_icon='cast', default_index=0,
                           orientation='vertical',
                           styles={'icon': {'color': 'orange', 'font-size': '16px'},
                                   "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                   "nav-link-selected": {"background-color": "green"},
                                   })

    "[View the source code](https://github.com/atonui/faqchatbot/tree/main)"


############################# Chat ######################################################
if selected == 'Chat':
    st.title('💬 FAQ Chatbot')
    # Create vector database if it doe not exist
    if not os.path.exists('faiss_db'):
        create_vector_db(FILEPATH)

    st.caption(':money_with_wings: Your friendly banking assistant.')
    
    # Initialise session state variables
    if 'messages' not in st.session_state:
        # st.session_state['messages'] = [{'role': 'assistant',
        #                                  'content': 'How can I help you today?'}]
        st.session_state.messages = []
        st.session_state['session_id'] = uuid.uuid4().hex

    # Display chat messages from history on app rerun
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])
        # with st.chat_message(msg['role']):
        #     st.markdown(msg['content'])

    # React to uset input
    if prompt := st.chat_input('How can I help you?'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        # Display user message
        st.chat_message('user').write(prompt)
        # st.write(st.session_state.messages)
        # Query LLM
        chain = get_qa_chain()
        response = chain(prompt)
        # Show user LLm response
        st.chat_message('ai').write(response['result'])
        # Store the user and LLM inputs in a list to keep session history
        # st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.session_state.messages.append({'role': 'ai', 'content': response['result']})

        # capture the promt, response and timestamp here in a sqlite database
        create_db_table()
        currentTime = datetime.datetime.now()
        insert_into_table(currentTime, st.session_state.session_id, prompt, response['result'])

############################# History ######################################################
if selected == 'History':
    st.title(':orange[Chat History]')
    def file_selector(folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames, key=filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    st.write('You selected `%s`' % filename)
    # create another tab where one can view these results
    df = read_db()
    st.dataframe(df, hide_index=True)

############################# About ######################################################
if selected == 'About':
    st.markdown(
        '''# Introduction
Companies all over the world are in constant communication with their clients to solve problems
that involve their products. After a while, some patterns will begin to emerge. Customer service
managers looking to make efficient use of their agents' time, will compile a list of Frequently
Asked Questions (FAQs) and put it up on their website. They will then have this as a first level of
support before the more complex queries can be shifted to a human agent. However, it doesn’t
always work out that way.
Why:
1. If you have many products or a big FAQ, no one will take their time to comb through it to
find the specific question related to their issue.
2. People like it when they talk to others or feel like they have talked to others.
Enter Large Language Models (LLMs). This project seeks to leverage the power of LLMs to
produce human-like natural responses to questions by fine-tuning an LLM using proprietary
data so that it can chat with customers and handle their most frequent queries. Customers can
interact with the model via chat, type their questions and have them answered immediately.
This system will be a huge time and resource saver for companies.

FAQ Chatbot is built with these core frameworks and modules:

- [**Streamlit**](https://streamlit.io/) - To create the web app UI and interactivity.
- [**Google PaLM**](https://ai.google/discover/palm2/) - LLM.
- [**Instructor Embeddings**](https://instructor-embedding.github.io/) - Used to create vector embeddings for the proprietary documents and the user queries.
- [**FAISS**](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) - Facebook AI Similarity Search, a vector database to store word embeddings.
- [**Langchain**](https://www.langchain.com/) - A Python Library for developing applications powered by LLM's.
- [**Dataset**](https://huggingface.co/datasets/clips/mfaq ) - Obtained from the Pivdenny bank FAQs via HuggingFace.

## 📈 **Future Roadmap**

Some potential features for future releases:

- User account system.
- Customise the prompt template and model hyperparameters
- Ability to create multiple knowledgebases.

## 📜 Current Prompt Template
Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section on the source document without making up anything.
        If the answer is not found in the context, kindly state "I do not know. Please contact customer care or visit one of our branches." Do not try to make up an answer.
        
        CONTEXT: {context}
    QUESTION: {question}

## :gear: Project Design
'''
    )

    st.image('project_design.png')

############################# Data ######################################################
if selected == 'Data':
    st.header('Proprietary Data')
    df = pd.read_csv(FILEPATH)
    st.dataframe(df)



# Add credit
st.sidebar.markdown('''
---
Built by [Allan Koech](https://github.com/atonui)
                    ''')
