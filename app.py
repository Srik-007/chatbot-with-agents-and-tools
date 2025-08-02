import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_key = os.getenv("HF_KEY")

# Initialize tools with proper configurations
apiwrap_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxi = ArxivQueryRun(api_wrapper=apiwrap_arxiv, name="Arxiv Search")

apiwrap_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=apiwrap_wiki, name="Wikipedia Search")

search = DuckDuckGoSearchRun(name="Web Search")

# Set up Streamlit page
st.set_page_config(
    page_title="AI Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "Hi, I'm a chatbot who can search the web, academic papers, and Wikipedia. How can I help you?"
        }
    ]

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
    index=0  # Default to first option
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
            temperature=0.3  # Added for more focused responses
        )
        
        # Set up tools
        tools = [search, arxi, wiki]
        
        # Initialize agent with better error handling
        search_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,  # Fixed parameter name
            verbose=True,
            max_iterations=5  # Prevent long-running operations
        )
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False
            )
            
            response = search_agent.run(
                {"input": prompt},  # Pass as dict for better compatibility
                callbacks=[st_cb]
            )
            
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.write(response)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"}
        )
