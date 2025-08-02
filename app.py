import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
import re
from dotenv import load_dotenv
def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
hf_key=os.getenv("HF_KEY")
apiwrap_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=500)
arxi=ArxivQueryRun(api_wrapper=apiwrap_arxiv)
apiwrap_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=apiwrap_wiki)
search=DuckDuckGoSearchRun(name="Search")
st.set_page_config(page_title="AI Chatbot", layout="wide",initial_sidebar_state="expanded")
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, i am a chatbot who can search the web. How can I help you?"}
    ]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
llm_model = st.sidebar.selectbox("Select a model", [
    "deepseek-r1-distill-llama-70b",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct"
])
if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    llm=ChatGroq(api_key=groq_api_key,model=llm_model,streaming=True)
    tools=[search,arxi,wiki]
    search_agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)


     
