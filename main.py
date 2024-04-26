import datetime
import os
import streamlit as st
from streamlit_option_menu import option_menu
from utils.functions import create_vector_db, \
    get_qa_chain, create_db_table, insert_into_table, read_db

st.set_page_config(page_title='FAQ Chatbot',
                   page_icon=':lips:', layout='centered')

with st.sidebar:
    selected = option_menu(None, ['Chat', 'History', 'About'],
                           icons=['chat-dots', 'clock-history'],
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
    def file_selector(folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    FILEPATH = 'banking.csv'
    create_vector_db(FILEPATH)

    st.caption(':money_with_wings: Your friendly banking assistant.')
    # Initialise session state variables
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
        st.chat_message('ai').write(response['result'])

        # capture the promt, response and timestamp here in a sqlite database
        create_db_table()
        currentTime = datetime.datetime.now()
        insert_into_table(currentTime, prompt, response['result'])

############################# History ######################################################
if selected == 'History':
    st.title(':orange[Chat History]')
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

## :gear: Project Design
'''
    )

    st.image('project_design.png')

# Add credit
st.sidebar.markdown("""
---
Built by [Allan Koech](https://github.com/atonui)""")
