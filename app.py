import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

# Optional: Load environment variables
load_dotenv()

# Initialize Streamlit app
st.title("🔎 LangChain - Chat with Search")

st.markdown("""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent
in an interactive Streamlit app.

Try more LangChain 🤝 Streamlit Agent examples at [LangChain Streamlit Agent](https://github.com/langchain-ai/streamlit-agent).
""")

# Sidebar input for Groq API Key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize Arxiv and Wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# DuckDuckGo search tool
search = DuckDuckGoSearchRun(name="Search")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user prompt
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM and tools
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    # Initialize LangChain agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Run agent with Streamlit callback
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        except Exception as e:
            response = "⚠️ An error occurred while searching. Please try again later."
            st.error(f"Error: {e}")
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)
