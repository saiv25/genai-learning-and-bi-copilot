
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found."
    )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
st.title("🤖 AskBuddy - QnA Bot")
st.markdown(
    """
    AskBuddy is a QnA bot that can answer your questions. 
    You can ask anything and it will try to answer you.
    """
)
if "messages" not in st.session_state:
    st.session_state.messages = st.session_state.get("messages", [])
 
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)
    
query=st.chat_input("Ask me anything:")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("assistant").markdown(res.content)
    st.session_state.messages.append({"role": "assistant", "content": res.content})
