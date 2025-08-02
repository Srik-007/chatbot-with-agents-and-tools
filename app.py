
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import re

# Clean up agent output
def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Setup search tools
apiwrap_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=apiwrap_arxiv, name="Arxiv Search")

apiwrap_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=apiwrap_wiki, name="Wikipedia Search")

search_tool = DuckDuckGoSearchRun(name="Web Search")

# Page configuration
st.set_page_config(page_title="AI Chatbot", layout="wide", initial_sidebar_state="expanded")

# Message & memory state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you?"}
    ]

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Model selector
llm_model = st.sidebar.selectbox("Select a model", [
    "deepseek-r1-distill-llama-70b",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct"
])

# Input and response handling
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    llm = ChatGroq(api_key=groq_api_key, model=llm_model, streaming=True)

    tools = [search_tool, arxiv_tool, wiki_tool]

    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=st.session_state["memory"],
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=5
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run(prompt, callbacks=[st_cb])
            response = clean_response(response)
        except Exception as e:
            response = f"Sorry, I encountered an error: {e}"

        st.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
