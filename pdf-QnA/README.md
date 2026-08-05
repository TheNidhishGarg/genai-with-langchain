# 📄 Chat with Your PDF – RAG Chatbot

An interactive **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, **Google Gemini**, and **ChromaDB**. Upload any PDF, ask questions in natural language, and receive answers generated solely from the document's content.

The application automatically creates a temporary vector database for every uploaded PDF, retrieves the most relevant document chunks, and uses **Gemini 2.5 Flash** to generate context-aware responses.

---

## 🚀 Features

- 📄 Upload any PDF directly through the web interface
- ✂️ Automatic document chunking using `RecursiveCharacterTextSplitter`
- 🧠 Semantic embeddings with **Gemini Embedding 001**
- 🗂️ Temporary in-memory **Chroma Vector Database**
- 🔍 Context retrieval using **Maximum Marginal Relevance (MMR)**
- 🤖 Question answering powered by **Gemini 2.5 Flash**
- 📚 View the retrieved chunks used to generate each answer
- 🎨 Clean and interactive Streamlit interface

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- ChromaDB
- PyPDFLoader
- python-dotenv

---

## 📂 Project Structure

```text
.
├── app.py              # Streamlit application
├── .env                # API Key
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

```text
             Upload PDF
                  │
                  ▼
          Temporary Storage
                  │
                  ▼
            Load PDF Pages
                  │
                  ▼
        Recursive Chunking
                  │
                  ▼
       Generate Embeddings
                  │
                  ▼
        Create Chroma Vector DB
                  │
                  ▼
        User Asks a Question
                  │
                  ▼
      Retrieve Relevant Chunks
                  │
                  ▼
         Build Context Prompt
                  │
                  ▼
        Gemini 2.5 Flash LLM
                  │
                  ▼
          Generate Response
```

---

## 📦 Installation

Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

Install the required packages

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💻 How to Use

1. Launch the Streamlit application.
2. Upload any PDF document.
3. Wait while the vector database is created.
4. Enter a question related to the document.
5. View the generated answer.
6. Expand **Retrieved Chunks** to inspect the document context used by the model.

---

## 🧠 RAG Pipeline

```text
PDF
 │
 ▼
PyPDFLoader
 │
 ▼
Chunking
 │
 ▼
Gemini Embeddings
 │
 ▼
Chroma Vector Store
 │
 ▼
MMR Retriever
 │
 ▼
Relevant Context
 │
 ▼
Gemini 2.5 Flash
 │
 ▼
Final Answer
```

---

## 📚 Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases
- Text Chunking
- Document Embeddings
- Prompt Engineering
- Large Language Models (LLMs)
- Streamlit Web Applications

---

## 🔮 Future Improvements

- Chat history and conversational memory
- Source page citations
- Multi-PDF support
- Persistent vector database
- Streaming LLM responses
- Hybrid (Keyword + Semantic) Retrieval
- Support for DOCX and TXT files

---

## 👨‍💻 Author

**Nidhish Garg**

If you found this project useful, consider giving the repository a ⭐.
