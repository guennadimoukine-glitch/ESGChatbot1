import streamlit as st
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredURLLoader,
    PlaywrightURLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ESG Report Research bot", layout="centered")
st.title("🤖 AI ESG Report Research Chatbot")

# Choose your LLM (Grok is free & fast)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # uncomment if you prefer OpenAI

@st.cache_resource
def load_and_index(_urls):
    docs = []

    # Local files
    if os.path.exists("data"):
        for file in os.listdir("data"):
            filepath = os.path.join("data", file)
            ext = os.path.splitext(file)[1].lower()
            if ext == ".pdf":
                loader = PyPDFDirectoryLoader("data")
                docs.extend(loader.load())
            elif ext == ".txt":
                loader = TextLoader(filepath, encoding="utf-8")
                docs.extend(loader.load())
            elif ext == ".docx":
                loader = Docx2txtLoader(filepath)
                docs.extend(loader.load())

    # Website crawling
    if _urls:
        with st.spinner(f"Crawling {len(_urls)} URL(s)... this may take 10-20s"):
            try:
                loader = PlaywrightURLLoader(urls=_urls, remove_selectors=["header", "footer", "nav"])
                docs.extend(loader.load())
            except:
                loader = UnstructuredURLLoader(urls=_urls)
                docs.extend(loader.load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
    return vectordb.as_retriever(search_kwargs={"k": 6})

# Sidebar
with st.sidebar:
    st.header("Data Source")
    url_input = st.text_input("Add URLs (one per line)", placeholder="https://yourcompany.com/help")
    urls = [u.strip() for u in url_input.split("\n") if u.strip()]

    if st.button("Re-index everything"):
        st.session_state.retriever = load_and_index(urls)
        st.success("Indexed!")

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

if "retriever" not in st.session_state:
    st.session_state.retriever = load_and_index([])

retriever = st.session_state.retriever
if not retriever:
    st.warning("Add files to `data/` folder or URLs above, then click Re-index.")
    st.stop()

def format_docs(docs):
    return "\n\n".join(f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content[:500]}..." for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent. Answer ONLY using the provided context. If you don't know, say so. Cite sources at the end."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda x: st.session_state.chat_history}
    | prompt
    | llm
    | StrOutputParser()
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("human").write(msg.content)
    else:
        st.chat_message("ai").write(msg.content)

if question := st.chat_input("Ask a question..."):
    st.chat_message("human").write(question)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = chain.invoke(question)
        st.write(response)
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=response))