import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

#model integration
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=2000,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="")]

st.title("Gemini Chatbot")

# Display previous conversation
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    if prompt.lower() == "clear":
        st.session_state.messages = [SystemMessage(content="")]
        st.rerun()

    # Show user message
    st.chat_message("user").write(prompt)

    # Save user message
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Get AI response
    response = model.invoke(st.session_state.messages)

    # Save AI response
    st.session_state.messages.append(AIMessage(content=response.content))

    # Display AI response
    st.chat_message("assistant").write(response.content)