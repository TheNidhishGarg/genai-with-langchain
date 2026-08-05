from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

#input question
#embed question
#retrieve data from vector store on basis of question
#ask the LLM about retrieved data and input question



#Loading all the things
embedding_model = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

vectorstore = Chroma(
    persist_directory="Chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs={
        "k": 4,
        "fetch_k":10,
        "lambda_mult": 0.5
    }
)

print("No. of chunks in db: ",vectorstore._collection.count())

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

#FLOW:

# LLM gets: question and context(which is the retrieved data from the vectorstore)
# question is from HUMAN
# context is from RETRIEVER


prompt = ChatPromptTemplate.from_messages(
    [
        ('system',
         """
        you are a helpful AI assistant.            

        use ONLY the provvided context to answer the question.
        if the answer is not present in the context,
        say: "I could not find the answer in the document."
        """),

        ('human',
        """
        {context}

        Question: {question}
        """)
    ]
)

print("***********RAG System created**********")
print("*\npress 0 to exit")

while True:
    query=input("You: ")
    if query=="0":
        break

    docs = retriever.invoke(query)   #This calls all the similar embeddings for query in docs
    

    # combining all the retrieved daata into one variable
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    print("\nRetrieved Chunks:\n")
    for i, doc in enumerate(docs, start=1):
        print(f"Chunk {i}")
        print(doc.page_content[:300])
        print("-"*50)

    #ChatPromptTemplate invoked
    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    })

    #response
    response = model.invoke(final_prompt)
    print(f"\nAI: {response.content}")
    


