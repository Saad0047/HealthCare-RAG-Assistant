import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import tempfile, os, base64

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Prompt for the agent to ensure it always uses the tool and doesn't rely on its own knowledge.
SYSTEM_PROMPT = """You are a medical document assistant. 
You MUST ALWAYS use the medical_pdf_search tool to look up information before answering ANY question.
NEVER answer from your own knowledge. ONLY answer based on what the tool returns.
If the tool returns nothing relevant, say 'I could not find this in the uploaded document.'"""

st.set_page_config(page_title="Healthcare RAG", layout="centered")
st.title("🏥 Healthcare Assistant")
st.caption("Upload a medical PDF or image and ask questions.")

uploaded = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    file_key = uploaded.name + str(uploaded.size)
    if st.session_state.get("file_key") != file_key:
        st.session_state["file_key"] = file_key
        st.session_state["agent"] = None
        st.session_state["file_type"] = uploaded.name.lower()
        st.session_state["file_bytes"] = uploaded.read()

        if st.session_state["file_type"].endswith(".pdf"):
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(st.session_state["file_bytes"])
                    tmp_path = f.name
                docs = PyPDFLoader(tmp_path).load()
                chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
                retriever = FAISS.from_documents(chunks, OpenAIEmbeddings()).as_retriever(
                    search_kwargs={"k": 6}
                )
                tool = create_retriever_tool(
                    retriever,
                    "medical_pdf_search",
                    "ALWAYS use this tool to search the uploaded PDF document. Input should be the user's question."
                )
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                st.session_state["agent"] = create_react_agent(llm, [tool], prompt=SYSTEM_PROMPT)
            st.success(f"✅ '{uploaded.name}' loaded and ready!")

question = st.text_input("Ask a question about the uploaded file")

if question:
    file_type = st.session_state.get("file_type", "")
    file_bytes = st.session_state.get("file_bytes", None)

    if file_type and file_type.endswith((".png", ".jpg", ".jpeg")) and file_bytes:
        with st.spinner("Analyzing image..."):
            mime = "image/png" if file_type.endswith(".png") else "image/jpeg"
            image_data = base64.b64encode(file_bytes).decode("utf-8")
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            response = llm.invoke([
                {"role": "system", "content": "You are a medical image analyst. Describe what you see in detail."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                    {"type": "text", "text": question}
                ]}
            ])
            st.write(response.content)

    elif st.session_state.get("agent"):
        with st.spinner("Searching document..."):
            result = st.session_state["agent"].invoke({
                "messages": [{"role": "user", "content": f"Search the uploaded PDF and answer: {question}"}]
            })
            st.write(result["messages"][-1].content)

    else:
        st.warning("Please upload a file first.")