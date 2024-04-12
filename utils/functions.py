import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from InstructorEmbedding import INSTRUCTOR
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
# from langchain.llms.google_palm import GooglePalm
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

load_dotenv()

llm = GoogleGenerativeAI(model="models/text-bison-001",
                         google_api_key=os.environ['GOOGLE_API_KEY'],
                         temperature=1)
# the temperature variable decides how creative the model can be, 0 is not and 1 is very

# create embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings()
vector_db_file_path = 'faiss_index'  # I wonder if we should hardcode this

@st.cache_resource
def create_vector_db(file_name):
    '''Create the vector db and save it in a local file rather than
    create it every time you run the app.'''
    loader = CSVLoader(file_path=file_name,
                       source_column='question',
                       encoding='UTF-8')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)
    vectordb.save_local(vector_db_file_path)

def get_qa_chain():
    # load the vector database from file
    vector_db = FAISS.load_local(vector_db_file_path,
                                 instructor_embeddings,
                                 allow_dangerous_deserialization=True)

    # create retriever for querying the vector db
    retriever = vector_db.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section on the source document without making up anything.
        If the answer is not found in the context, kindly state "I do not know." Do not try to make up an answer.
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

if __name__ == '__main__':

    # Create a database if one does not exist
    # if not os.path.isdir(vector_db_file_path):
    #     create_vector_db()

    chain = get_qa_chain()

    # print(chain('Do you provide internship?'))
