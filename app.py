import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import re

# === Cleaning function ===
def clean_response(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)

# === Load environment variables ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# === Setup Streamlit page ===
st.set_page_config(
    page_title="AI Chatbot with Memory", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Initialize chat message history ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hi, I'm a chatbot who can search the web, acad
