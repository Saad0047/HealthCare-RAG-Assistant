# 🏥 Healthcare AI Assistant

An AI-powered Healthcare Assistant built using Streamlit, LangChain, FAISS, and OpenAI GPT-4o that allows users to upload medical PDFs or images and ask questions in natural language.

The system uses a Retrieval-Augmented Generation (RAG) pipeline for document-based question answering and integrates GPT-4o Vision for medical image analysis.

---

# 🚀 Features

- 📄 Upload and analyze medical PDF documents
- 🖼️ Upload and analyze medical images
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic search using FAISS vector database
- 🤖 LangChain + LangGraph agent orchestration
- 👁️ GPT-4o Vision integration for image understanding
- 💬 Natural language question answering
- ⚡ Streamlit-based interactive UI

---

# 🏗️ Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- OpenAI API
- FAISS
- GPT-4o / GPT-4o-mini
- PyPDFLoader
- RecursiveCharacterTextSplitter

---

# 📂 Project Workflow

## PDF Flow
1. User uploads a medical PDF
2. PDF is loaded and parsed
3. Text is split into chunks
4. Embeddings are generated using OpenAI Embeddings
5. Chunks are stored in FAISS vector database
6. Relevant chunks are retrieved based on user query
7. LLM generates a context-aware response

## Image Flow
1. User uploads a medical image
2. Image is converted to Base64 format
3. GPT-4o Vision analyzes the image
4. AI responds to user questions about the image

---

# 🔐 Security Notes

This project uses environment variables for API key security.

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Saad0047/HealthCare-RAG-Assistant.git
cd HealthCare-RAG-Assistant
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Add Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## 4. Run the Application

```bash
streamlit run Healthcare_RAG_bot.py
```

---

# 🧠 Key AI Concepts Used

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Multimodal AI
- LLM Agents
- Context-Aware Response Generation

---

# 📸 Future Improvements

- User Authentication
- Cloud Deployment
- Multi-document support
- Medical knowledge validation layer
- Conversation history memory
- Advanced UI/UX improvements

---

# 👨‍💻 Author

Developed by Muhammad Saad Ullah — passionate about AI, software development, and building real-world intelligent systems.

---

# ⭐ Acknowledgements

- OpenAI
- LangChain
- Streamlit
- FAISS
