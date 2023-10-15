from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]
st.write(f'{openai.api_base}')
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key=openai.api_key,  
                        openai_organization= openai.api_type ,
                        openai_api_base = openai.api_base,
                        temperature=0,
                        model_name = MODEL_NAME,
                        engine=MODEL_NAME,
                         streaming=True, callbacks=[stream_handler])
        response = llm(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
