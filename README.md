# 🏥 Medical QA RAG System

An **AI-powered Medical Question Answering Chatbot** built using **Retrieval-Augmented Generation (RAG)**. The system retrieves relevant medical knowledge from documents and generates accurate, context-aware answers using a large language model.

Designed as a **production-style project**, this repository demonstrates practical experience with modern AI stacks including **LangChain, Pinecone, Groq LLMs, Flask and conversation memory buffer**.

---

## 🚀 Features

* 📄 **Document-based Medical QA** using RAG
* 🔍 **Semantic search** with Pinecone Vector Database
* 🤖 **LLM-powered answers** using Groq (LLaMA 3)
* 🧠 **HuggingFace embeddings** for document vectorization
* 🌐 **Flask web interface** for real-time chat
* 🧠 **Conversation memory buffer** to handle follow-ups, pronouns, and topic switching 
* 🔐 Environment-based API key management

---

## 🛠️ Tech Stack

| Category    | Tools            |
| ----------- | ---------------- |
| Language    | Python           |
| LLM         | Groq (LLaMA 3.1) |
| Framework   | LangChain        |
| Vector DB   | Pinecone         |
| Embeddings  | HuggingFace      |
| Backend     | Flask            |
| Frontend    | HTML, CSS        |
| Environment | Conda / venv     |

---

## 📂 Project Structure

```
medical-QA-RAG-system/
│
├── app.py                 # Flask application entry point
├── store_index.py         # Script to create & store embeddings in Pinecone
├── requirements.txt       # Project dependencies
├── setup.py               # Package configuration
├── .env                   # Environment variables (ignored in git)
├── .gitignore
├── README.md
│
├── data/
│   └── Medical_book.pdf   # Source medical document
│
├── src/
│   ├── helper.py          # Embedding & document loading logic
│   ├── prompt.py          # System prompt for medical assistant
│   └── __init__.py
│
├── templates/
│   └── chat.html          # Chat UI
│
├── static/
│   └── style.css          # Styling
│
└── research/
    └── trials.ipynb       # Experiments & testing
```

---

## 🧩 Architecture Overview

1. **Document Ingestion** → PDF loaded & chunked
2. **Embeddings** → HuggingFace embeddings generated
3. **Vector Store** → Stored & searched via Pinecone
4. **Retrieval** → Top-k relevant chunks fetched
5. **Conversation Memory Buffer** → Tracks user conversation context, handles follow-ups, pronouns, and topic switches
6. **Generation** → Groq LLaMA 3 produces grounded answers
7. **API/UI** → Flask serves responses to a chat UI

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/medical-QA-RAG-system.git
cd medical-QA-RAG-system
```

### 2️⃣ Create & activate environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

---

## 📥 Index Medical Documents

Run once to store embeddings in Pinecone:

```bash
python store_index.py
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open in browser:
👉 `http://localhost:8080`

---

## 💬 Example Workflow

1. User asks a medical question
2. Query is embedded using HuggingFace
3. Relevant medical chunks retrieved from Pinecone
4. Conversation memory buffer maintains context for follow-up questions
5. Context + query passed to Groq LLM
6. Accurate, grounded response returned to user

---

## ⚠️ Medical Disclaimer

This chatbot is for **educational purposes only** and does **not** replace professional medical advice.

---

## 📌 Resume Highlights

* Built an **end-to-end Medical RAG system** using LangChain
* Integrated **vector search (Pinecone)** with LLM inference
* Designed **Flask-based AI API & UI**
* Implemented **conversation memory buffer** for follow-ups and context
* Designed **scalable document ingestion pipeline**
* Followed **production best practices** (env vars, modular code)

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Rajnish Gupta**
Software Engineer | AI & Data Engineering Enthusiast

---

⭐ If you found this project useful, consider giving it a star!
