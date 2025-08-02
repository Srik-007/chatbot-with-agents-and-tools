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
            "content": "Hi, I'm a chatbot who can search the web, academic papers, and Wikipedia. How can I help you?"
        }
    ]

# === Initialize memory ===
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# === Display previous chat messages ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# === Sidebar: Model selection ===
llm_model = st.sidebar.selectbox(
    "Select a model",
    options=[
        "deepseek-r1-distill-llama-70b",
        "moonshotai/kimi-k2-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ],
    index=0
)

# === Initialize tools ===
arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500),
    name="Arxiv Search"
)
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500),
    name="Wikipedia Search"
)
search_tool = DuckDuckGoSearchRun(name="Web Search")

tools = [search_tool, arxiv_tool, wiki_tool]

# === Handle user input ===
if prompt := st.chat_input(placeholder="Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        # === Initialize LLM ===
        llm = ChatGroq(
            api_key=groq_api_key,
            model=llm_model,
            streaming=True,
            temperature=0.3
        )

        # === Initialize agent with memory ===
        search_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,
            memory=st.session_state.memory,
            max_iterations=5
        )

        # === Generate and show response ===
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = search_agent.run(prompt, callbacks=[st_cb])
            response = clean_response(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# === (Optional) Show memory debug info ===
with st.sidebar.expander("🧠 Conversation Memory", expanded=False):
    for m in st.session_state.memory.chat_memory.messages:
        st.write(f"**{m.type.capitalize()}**: {m.content}")
