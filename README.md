# 🩺 VruCare: Kidney Stone Information Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kidneystone-rag-chatbot.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://python.langchain.com/)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-purple)](https://openrouter.ai/)

**VruCare** is an advanced Retrieval-Augmented Generation (RAG) chatbot designed to provide accurate, safe, and sourced information about kidney stones. It combines semantic search with keyword matching using a custom **Reciprocal Rank Fusion (RRF)** algorithm to deliver highly relevant answers from a curated medical knowledge base.

---

## 🚀 Key Features

* **🧠 Hybrid Search Engine:** Uses a custom-built retriever that combines **FAISS** (semantic search) and **BM25** (keyword search) to find the most relevant medical documents.
* **🏆 Reciprocal Rank Fusion (RRF):** Implements a sophisticated re-ranking algorithm to fuse results from both search methods, ensuring high accuracy for both broad concepts and specific medical terms (e.g., "Tamsulosin").
* **💬 Context-Aware Conversations:** Remembers chat history and intelligently reformulates follow-up questions (e.g., understanding "What are the side effects?" refers to the previous topic).
* **⚡ Streaming Responses:** Provides a real-time, typewriter-style experience for low-latency interaction.
* **🛡️ Safety First:** Engineered with strict system prompts to refuse medical diagnosis or treatment advice, directing users to professionals.
* **🆓 Cost-Effective:** Powered by **OpenRouter's free tier** (Gemini 2.0 Flash / Llama 3), making it free to run while maintaining high performance.

---

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **Orchestration:** LangChain (Core, Community, OpenAI)
* **LLM Provider:** OpenRouter (Gemini 2.0 Flash / Llama 3)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Keyword Search:** Rank-BM25

---

## 📂 Project Structure

```text
Kidneystone-rag-chatbot/
├── data/                  # Folder containing raw PDF medical documents
├── src/
│   ├── app.py             # Main Streamlit application (UI & Chat Logic)
│   ├── rag_pipeline.py    # Core RAG logic (Custom Hybrid Retriever & RRF)
│   └── vectorizer.py      # Script to ingest PDFs and build the vector DB
├── vectorstore/           # Generated FAISS index and raw text chunks
├── .env                   # API keys (ignored by git)
├── .gitignore             # Files to exclude from git
├── requirements.txt       # Python dependencies for pip
└── README.md              # Project documentation

```
----

## ⚙️ Installation & Setup

### 1. Clone the Repository
```
bash
git clone [https://github.com/Karthikrishna05/Kidneystone-rag-chatbot.git](https://github.com/Karthikrishna05/Kidneystone-rag-chatbot.git)
cd Kidneystone-rag-chatbot


```
2. Create a Virtual Environment
```
Bash

# On Windows:
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```
3. Install Dependencies
```
Bash

pip install -r requirements.txt
```
4. Configure API Keys
Create a .env file in the root directory and add your OpenRouter API key:
```

OPENROUTER_API_KEY="sk-or-v1-..."
```
🏃‍♂️ How to Run
Step 1: Build the Knowledge Base ("The Brain")
Before running the app, you must ingest your PDF documents. Place your PDFs in the data/ folder and run:
```
Bash

python src/vectorizer.py
```
This will create the vectorstore/ folder containing your index.

Step 2: Launch the Chatbot
```
Bash

streamlit run src/app.py
```
The app will open in your browser at http://localhost:8501.

## 📚 Knowledge Base Sources
The chatbot's intelligence is grounded strictly in the following authoritative texts:
* **American Urological Association (AUA):** Medical Management Guidelines[cite: 1076].
* **National Kidney Foundation:** Diet and Stone Prevention Guides[cite: 1].
* **NIDDK:** Urologic Diseases in America Project Data[cite: 1701].
* **Recent Research:** Systematic reviews on fluid intake and prevention[cite: 2142].
