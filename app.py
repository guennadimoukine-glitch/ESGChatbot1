import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader, Docx2txtLoader, UnstructuredURLLoader
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

st.set_page_config(page_title="AI ESG Bot", layout="centered")
st.title("AI ESG Bot")
st.caption("Get data from ESG reports")

# ============================
# HARD-CODED URLS (edit here)
# ============================

DEFAULT_URLS = [
    "https://www.crowdstrike.com/about/environmental-social-governance/",
    "https://www.paloaltonetworks.com/about-us/corporate-responsibility",
    "https://www.fortinet.com/corporate/about-us/corporate-social-responsibility/sustainability-report",
    "https://www.veeam.com/company/corporate-governance.html",
    "https://www.commvault.com/document/commvault-fy25-sustainability-report",
    "https://www.cohesity.com/company/sustainability/",
    "https://www.zscaler.com/corporate-responsibility",
    "https://www.cyberark.com/company/esg/",
    "https://www.trendmicro.com/en_us/about/sustainability.html",
    "https://www.microsoft.com/en-us/corporate-responsibility/reports-hub",
    "https://www.cisco.com/c/en/us/about/csr/esg-hub.html",
    "https://www.broadcom.com/company/corporate-responsibility",
    "https://www.dell.com/en-us/lp/dt/reports-and-resources",
    "https://www.ibm.com/impact",
    "https://www.acronis.com/en/sustainability-governance/",
    # Add as many as you want → they will always be indexed automatically
]

llm = ChatXAI(model="grok-beta", temperature=0)

@st.cache_resource
def get_retriever(urls):
    docs = []

    # Local PDFs/txt/docx
    if os.path.exists("data"):
        for f in os.listdir("data"):
            p = os.path.join("data", f)
            ext = os.path.splitext(f)[1].lower()
            if ext == ".pdf": docs.extend(PyPDFDirectoryLoader("data").load())
            if ext == ".txt": docs.extend(TextLoader(p).load())
            if ext == ".docx": docs.extend(Docx2txtLoader(p).load())

    # URLs
    if urls:
        with st.spinner(f"Loading {len(urls)} URLs…"):
            loader = UnstructuredURLLoader(urls=urls, mode="single")
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    
    # THIS IS THE MAGIC LINE — best of both worlds
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20, "fetch_k": 50})

# Sidebar (unchanged)
with st.sidebar:
    st.header("Knowledge Base")
    for u in DEFAULT_URLS: st.write(f"✓ {u}")
    extra = st.text_area("Add more URLs (optional)", height=100)
    extra_urls = [l.strip() for l in extra.split("\n") if l.strip()]
    all_urls = DEFAULT_URLS + extra_urls

    if st.button("Re-index everything", type="primary"):
        st.session_state.retriever = get_retriever(all_urls)
        st.success("Done!")

    if st.button("Clear chat"): st.session_state.chat_history = []; st.rerun()

# Init
if "retriever" not in st.session_state:
    with st.spinner("Loading default knowledge base…"):
        st.session_state.retriever = get_retriever(DEFAULT_URLS)
if "chat_history" not in st.session_state: st.session_state.chat_history = []

retriever = st.session_state.retriever

# PERFECT PROMPT — forces long, detailed, sourced answers
system_prompt = """You are a sustainability expert with complete access to the sustainability program and ESG initiatives of multiple companies. Your role is to compare them.

Answer in full paragraphs with as much detail as possible. 
Use bullet points or numbered lists when it makes the answer clearer.
Never give short answers — always be thorough and helpful.
If multiple sources contain relevant info, combine them. Where possible, use all sources.
Always end with the exact sources used.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# This avoids the Groq token-limit crash
current_history = st.session_state.chat_history[-10:]  # last 10 messages only → safe

def format_docs(docs):
    return "\n\n".join(f"Source → {d.metadata.get('source','Unknown')}\n{d.page_content}" for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda x: current_history}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat UI
for m in st.session_state.chat_history:
    if isinstance(m, HumanMessage):
        st.chat_message("human").write(m.content)
    else:
        st.chat_message("ai").write(m.content)

if question := st.chat_input("Ask me anything…"):
    st.chat_message("human").write(question)
    with st.chat_message("ai"):
        with st.spinner("Generating detailed answer…"):
            answer = chain.invoke(question)
        st.markdown(answer)

    st.session_state.chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])