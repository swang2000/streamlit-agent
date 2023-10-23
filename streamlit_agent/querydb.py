import streamlit as st
from pathlib import Path
import pandas as pd, os
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit


load_dotenv()
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
EMBEDDING_API_BASE = os.getenv('EMBEDDING_API_BASE')
EMBEDDING_API_VERSION = os.getenv('EMBEDDING_API_VERSION')
EMDEDDING_ENGINE = os.getenv('EMDEDDING_ENGINE')


#ChatGPT credentials
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_deployment_name = os.getenv('OPENAI_DEPLOYMENT_NAME')
openai_embedding_model_name = os.getenv('OPENAI_EMBEDDING_MODEL_NAME')
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version =  os.getenv('OPENAI_API_VERSION')
MODEL_NAME = os.getenv('MODEL_NAME')


st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# User inputs
# radio_opt = ["Use sample database - Chinook.db", "Connect to your SQL database"]
# selected_opt = st.sidebar.radio(label="Choose suitable option", options=radio_opt)
# if radio_opt.index(selected_opt) == 1:
#     db_uri = st.sidebar.text_input(
#         label="Database URI", placeholder="mysql://user:pass@hostname:port/db"
#     )
# else:
#     db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
#     db_uri = f"sqlite:////{db_filepath}"

# openai_api_key = st.sidebar.text_input(
#     label="OpenAI API Key",
#     type="password",
# )

import sqlalchemy as sa
from sqlalchemy.engine.url import URL

# build the sqlalchemy URL
# url = URL.create(
# drivername='redshift+redshift_connector', # indicate redshift_connector driver and dialect will be used
# host='itx-ags-prd-rs-cl-01.cku868xglwj7.us-east-1.redshift.amazonaws.com', # Amazon Redshift host
# port=5439, # Amazon Redshift port
# database='cdeprddb', # Amazon Redshift database
# username='eureka_prd_read', # Amazon Redshift username
# password='=1j-uL9xpOfF' # Amazon Redshift password
# )

url = URL.create(
drivername='mysql+pymysql', # indicate redshift_connector driver and dialect will be used
host='itx-acm-jsa-mdm-dev.czijpxum5el7.us-east-1.rds.amazonaws.com', # Amazon Redshift host
port=3306, # Amazon Redshift port
database='jsa_poc', # Amazon Redshift database
username='jsa_poc_app', # Amazon Redshift username
password='X4o6bUsx-uQI' # Amazon Redshift password
)



# engine = sa.create_engine(url)
# df = pd.read_sql('''select * from praada_act_d limit 10''', engine)

# Check user inputs
# if not db_uri:
#     st.info("Please enter database URI to connect to your database.")
#     st.stop()

# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()

# Setup agent
# llm = OpenAI(openai_api_key=openai.api_key, temperature=0, streaming=True)
llm = ChatOpenAI(
            model_name="gpt-35-turbo-16k", engine = MODEL_NAME, openai_api_key=openai.api_key, temperature=0, streaming=True
        )

@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri)


db = configure_db(url)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
