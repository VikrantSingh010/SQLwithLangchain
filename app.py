import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import pymysql

# Page configuration
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜", layout="wide")

# Custom CSS to improve UI appearance
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: "Roboto", sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #002b36;
        color: #ffffff;
    }
    h1, h3, h4 {
        color: #004466;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🦜 LangChain: Chat with SQL Database")
st.write("""
Welcome to the **LangChain SQL Chat**! This app allows you to query your SQL database directly 
using natural language via a powerful language model.
""")

# Sidebar options for DB selection
radio_opt = ["Use SQLite 3 Database (student.db)", "Connect to MySQL Database"]
selected_opt = st.sidebar.radio("Choose the DB you want to chat with", options=radio_opt)

# MySQL connection inputs
if radio_opt.index(selected_opt) == 1:
    db_uri = "USE_MYSQL"
    st.sidebar.subheader("MySQL Connection Details")
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = "USE_LOCALDB"

# Developer and API key section
st.sidebar.subheader("Developer Information")
with st.sidebar.expander("Show developer details"):
    if st.checkbox("Show developer details"):
        st.write("**Name:** Vikrant Singh")
        st.write("**Email:** b22Ai043@iitj.ac.in")

st.sidebar.write("""
- [LangChain Documentation](https://docs.langchain.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Groq API](https://api.groq.com)
""")
api_key = st.sidebar.text_input("Groq API Key", type="password")

# Inform the user if input is incomplete
if not db_uri:
    st.info("Please enter the database information and uri")
elif not api_key:
    st.info("Please provide the Groq API key")

# Configure the database connection
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == "USE_LOCALDB":
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == "USE_MYSQL":
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        try:
            db = SQLDatabase(create_engine(
                f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            ))
            st.success(f"Successfully connected to MySQL database: {mysql_db}")
            return db
        except Exception as e:
            st.error(f"Error connecting to MySQL: {e}")
            st.stop()

# Set up LLM and database
if db_uri and api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    if db_uri == "USE_MYSQL":
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    else:
        db = configure_db(db_uri)

    # Toolkit setup
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create SQL agent
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # Chat history management
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with your database today?"}]

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User query input
    user_query = st.chat_input(placeholder="Ask anything from the database...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
else:
    st.warning("Please provide all required inputs to proceed.")
