# here we will only ready up our database
#loading, chunking, embeddings and storing would be done by here!


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv
load_dotenv()

#Loading Document
data = PyPDFLoader("rag_project1/document/AttentionIsAllYouNeed.pdf")
docs=data.load()

#Chunking Document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)

print(f"Total pages: {len(docs)}")
print(f"Total chunks: {len(chunks)}")

#Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001",
    dimension=512,
    api_key=os.getenv("GOOGLE_API_KEY")
)


#Embedding the Chunks & Storing in vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="Chroma_db"
)

print("Database created successfully!")
print("Vectors stored:", vectorstore._collection.count())