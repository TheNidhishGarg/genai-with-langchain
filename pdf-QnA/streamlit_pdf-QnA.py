import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="PDF RAG", page_icon="📄")

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Creating Vector Database..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        embedding_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=os.getenv("GOOGLE_API_KEY"),
            dimension=512
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:
'I could not find the answer in the document.'
"""
                ),
                (
                    "human",
                    """
{context}

Question: {question}
"""
                )
            ]
        )

    st.success("Database Created Successfully!")

    question = st.text_input("Ask a question from the PDF")

    if st.button("Ask"):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = model.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(docs, 1):
                st.write(f"### Chunk {i}")
                st.write(doc.page_content)
                st.divider()

    os.remove(pdf_path)