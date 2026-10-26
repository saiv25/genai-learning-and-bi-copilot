from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b", streaming =True)

search = GoogleSerperAPIWrapper()
tools = [search.run]
if"memory" not in st.session_state:
     st.session_state.memory = InMemorySaver()
     st.session_state.history = []


agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer= st.session_state.memory,
    system_prompt="You are a helpful assistant that answers questions based on the context provided. If you don't know the answer, just say you don't know. Do not try to make up an answer.",
    )
print(st.session_state.memory)

st.title("QuickAnswer Bot with Groq")
for  message in st.session_state.history:
     role = message["role"]
     content = message["content"]
     st.chat_message(role).markdown(content)
     
query=st.chat_input("Ask me anything!")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    response = agent.stream({"messages": [{"role": "user", "content": query}]},
                      {"configurable":{"thread_id": "1"}},stream_mode="messages")
    answer = response["messages"][-1].content
    st.chat_message("assistant").markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})





