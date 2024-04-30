import os
import sqlite3
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import pandas as pd

load_dotenv()

llm = GoogleGenerativeAI(model="models/text-bison-001",
                         google_api_key=os.environ['GOOGLE_API_KEY'],
                         temperature=1)
# the temperature variable decides how creative the model can be, 0 is not and 1 is very

# create embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings()
vector_db_file_path = 'faiss_db'

@st.cache_resource
def create_vector_db(file_path):
    '''
    Create the vector db and save it in a local file rather than
    create it every time you run the app.

    Args:
    filepath: Name of the csv file to be vectorised.
    '''
    loader = CSVLoader(file_path=file_path,
                       source_column='question',
                       encoding='UTF-8')
    data = loader.load()
    st.success('Loader okay')
    try:
        vectordb = FAISS.afrom_documents(
            documents=data,
            embedding=instructor_embeddings
            )
        st.success('DB created but not saved locally')
    except:
        st.warning('DB creation failed')
    vectordb.save_local(vector_db_file_path)


def get_qa_chain():
    '''
    Function to take the user query and relevant chunk of data
    from the vector database and prompt template, pass it to the LLM
    and return the response.

    Returns: Chain object
    '''
    # load the vector database from file
    vector_db = FAISS.load_local(vector_db_file_path,
                                 instructor_embeddings,
                                 allow_dangerous_deserialization=True)

    # create retriever for querying the vector db
    retriever = vector_db.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section on the source document without making up anything.
        If the answer is not found in the context, kindly state "I do not know. Please contact customer care or visit one of our branches." Do not try to make up an answer.
        CONTEXT: {context}
        QUESTION: {question}
        """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=['context', 'question']
    )
    chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            input_key='query',
            return_source_documents=True,
            chain_type_kwargs={'prompt': PROMPT}
            )
    return chain


def open_db_connection():
    ''''
    Open db connection and return connection object.

    Returns a database connection object.
    '''
    db_connection = sqlite3.connect('chat.db',
                                    detect_types=sqlite3.PARSE_DECLTYPES |
                                    sqlite3.PARSE_COLNAMES)
    return db_connection


def close_db_connection(cursor):
    '''
    Close database connection

    Args:
    cursor: database cursor object.
    '''
    cursor.close()


def create_db_table():
    ''''
    Create databsase table chat_table
    '''
    cursor = open_db_connection().cursor()
    # create table if not found
    create_table_query = '''CREATE TABLE IF NOT EXISTS chat_table (
    ID INTEGER PRIMARY KEY,
    time_stamp TIMESTAMP,
    user_query VARCHAR,
    llm_response VARCHAR
    );'''

    cursor.execute(create_table_query)
    close_db_connection(cursor)


def insert_into_table(current_time, prompt, response):
    '''
    Insert values into the chat_table table in the database
    '''
    db_connection = open_db_connection()
    cursor = db_connection.cursor()
    cursor.execute('''INSERT INTO chat_table (time_stamp, user_query, llm_response) VALUES(?,?,?)''',
                   [current_time, prompt, response])
    db_connection.commit()
    close_db_connection(cursor)


def read_db():
    '''
    Read all values from the table and return them as a dataframe.
    '''
    db_connection = open_db_connection()
    cursor = db_connection.cursor()
    df = pd.read_sql_query('''SELECT * FROM chat_table''', db_connection)
    df.reset_index(drop=True, inplace=True)
    close_db_connection(cursor)
    return df
