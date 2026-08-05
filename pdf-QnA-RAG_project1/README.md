# 📄 Basic RAG (Retrieval-Augmented Generation) with LangChain & Gemini

A simple implementation of a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask questions about a PDF document using **Google Gemini**, **LangChain**, and **ChromaDB**.

This project demonstrates the core workflow behind modern RAG systems—from loading and embedding a document to retrieving relevant context and generating accurate answers using an LLM.

---

## 🚀 Features

- Load PDF documents using **PyPDFLoader**
- Split documents into semantic chunks
- Generate embeddings using **Google Gemini Embedding Model**
- Store embeddings in **Chroma Vector Database**
- Retrieve relevant chunks using **MMR (Max Marginal Relevance) Search**
- Answer user queries using **Gemini 2.5 Flash**
- Simple command-line interface for interactive Q&A

---

## 🛠️ Tech Stack

- Python
- LangChain
- Google Gemini API
- ChromaDB
- PyPDFLoader
- python-dotenv

---

## 📂 Project Structure

```text
.
├── create_database.py      # Creates the vector database
├── main.py                 # Runs the RAG question-answering pipeline
├── document/
│   └── AttentionIsAllYouNeed.pdf
│
├── .env
└── README.md
```

---

## ⚙️ Workflow

```text
                create_database.py

        PDF
         │
         ▼
  Load Document
         │
         ▼
     Chunking
         │
         ▼
 Generate Embeddings
         │
         ▼
 Store in ChromaDB


                  main.py

     User Question
           │
           ▼
   Retrieve Relevant Chunks
           │
           ▼
     Build Context
           │
           ▼
     Prompt Gemini
           │
           ▼
     Generate Answer
```

---

## 📦 Installation

Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

Install the dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

---

## ▶️ Usage

### Step 1: Create the Vector Database

```bash
python create_database.py
```

This script:

- Loads the PDF
- Splits it into chunks
- Creates embeddings
- Stores them in ChromaDB

---

### Step 2: Start the RAG Chat

```bash
python main.py
```

Example:

```text
***********RAG System created**********

You:
What is the Transformer architecture?

AI:
The Transformer is a sequence transduction model based entirely on attention mechanisms...
```

---

## 📚 Core Concepts Used

- Retrieval-Augmented Generation (RAG)
- Document Loading
- Recursive Text Splitting
- Vector Embeddings
- Vector Databases
- Maximum Marginal Relevance (MMR)
- Semantic Search
- Prompt Engineering

---

## 📈 Future Improvements

- Streamlit Web Interface
- PDF Upload Support
- Chat History / Memory
- Source Citations
- Multi-PDF Support
- Persistent Conversation
- Hybrid Search (Keyword + Semantic)

---

## 👨‍💻 Author

**Nidhish Garg**

If you found this project helpful, consider giving the repository a ⭐.
