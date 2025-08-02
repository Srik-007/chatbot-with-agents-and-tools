
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# Basic setup
st.set_page_config(page_title="AI Chatbot with Memory", layout="wide")
st.title("🤖 Chatbot with Memory")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# LLM and tools setup
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
search = DuckDuckGoSearchRun()
tools = [Tool(name="DuckDuckGo Search", func=search.run, description="Search the web")]

# Agent setup
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# User input
if prompt := st.chat_input("Ask me something..."):
    # Store user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show AI response container
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())

        # Run agent with full context
        full_context = "\n".join([m.content for m in st.session_state.messages])
        response = agent_executor.run(full_context, callbacks=[st_cb])

        st.markdown(response)
        st.session_state.messages.append(AIMessage(content=response))
