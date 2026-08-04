from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv


load_dotenv()

model=ChatGoogleGenerativeAI(
model="gemini-2.5-flash",
api_key=os.getenv("GOOGLE_API_KEY"),
temperature=0,
max_tokens = 2000)


messages =[SystemMessage(content="")]

print("------Welcome! How can I assist you today?------")

while True:
    prompt = input("You :")
    messages.append(HumanMessage(content=prompt))
    
    if prompt=="clear" :
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot :", response.content)

print(messages)