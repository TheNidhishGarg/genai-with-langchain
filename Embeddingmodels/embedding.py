from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv;
import os

load_dotenv();

api_key = os.getenv("GOOGLE_API_KEY")
print(api_key)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    dimension=50,
    google_api_key=api_key
)
query=input("Enter your query :")
result = embeddings.embed_query(query)
print(result)

