from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader

import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_API_KEY")
)

url = "https://www.appnwebtechnologies.com/blog/decoding-the-rapido-app-a-comprehensive-framework-breakdown"

data = WebBaseLoader(url)#Creates a loader for the text file
docs = data.load()#reads the file and return list of document objectss


template = ChatPromptTemplate.from_messages([
    ('system','you are an AI summariser, Summarize this text in no more than 150-200words'),
    ('human','{data}')
])

prompt = template.format_messages(data=docs[0].page_content)

response=model.invoke(prompt)
print("Heres the Summary of your Paragraph : \n\n")
print(response.content)