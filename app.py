import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferWindowMemory  # <-- Add memory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize tools
apiwrap_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxi = ArxivQueryRun(api_wrapper=apiwrap_arxiv, name="Arxiv Search")
apiwrap_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=apiwrap_wiki, name="Wikipedia Search")
search = DuckDuckGoSearchRun(name="Web Search")

# Set up Streamlit page
st.set_page_config(
    page_title="AI Chatbot with Memory", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hi, I'm a chatbot who can search the web, academic papers, and Wikipedia. How can I help you?"
        }
    ]
    
if "memory" not in st.session_state:
    # Keep last 5 messages in memory
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=2000000,
        return_messages=True
    )

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Model selection in sidebar
llm_model = st.sidebar.selectbox(
    "Select a model",
    options=[
        "deepseek-r1-distill-llama-70b",
        "moonshotai/kimi-k2-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ],
    index=0
)

# Chat input and processing
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    try:
        # Initialize LLM
        llm = ChatGroq(
            api_key=groq_api_key,
            model=llm_model,
            streaming=True,
        )
        
        # Set up tools
        tools = [search, arxi, wiki]
        
        # Initialize agent with memory
        search_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,
            memory=st.session_state.memory,  # Add memory to agent
            agent_kwargs={
                "input_variables": ["input", "chat_history", "agent_scratchpad"]
            }
        )
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=True
            )
            
            response = search_agent.run(
                {
                    "input": prompt,
                    "chat_history": st.session_state.memory.buffer  # Pass chat history
                },
                callbacks=[st_cb]
            )
            
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.write(response)
