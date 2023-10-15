import os
import tempfile
#import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os



EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
EMBEDDING_API_BASE = os.getenv('EMBEDDING_API_BASE')
EMBEDDING_API_VERSION = os.getenv('EMBEDDING_API_VERSION')
EMDEDDING_ENGINE = os.getenv('EMDEDDING_ENGINE')


#ChatGPT credentials
import openai
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai_deployment_name = os.getenv('OPENAI_DEPLOYMENT_NAME')
openai_embedding_model_name = os.getenv('OPENAI_EMBEDDING_MODEL_NAME')
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version =  os.getenv('OPENAI_API_VERSION')
MODEL_NAME = os.getenv('MODEL_NAME')


loader = PyPDFLoader('/Users/swang294/Library/CloudStorage/OneDrive-JNJ/Projects/Doc comparison/AHFS STELARA (Ustekinumab) - TDS Health.pdf')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

from PyPDF2 import PdfReader

reader = PdfReader("/Users/swang294/Library/CloudStorage/OneDrive-JNJ/Projects/Doc comparison/AHFS STELARA (Ustekinumab) - TDS Health.pdf")
page = reader.pages[1]
print(page.extract_text())

from langchain.document_loaders import PDFMinerLoader

loader = PDFMinerLoader("/Users/swang294/Library/CloudStorage/OneDrive-JNJ/Projects/Doc comparison/AHFS STELARA (Ustekinumab) - TDS Health.pdf")

data = loader.load()
print(data[0].page_content)