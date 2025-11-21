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

st.set_page_config(page_title="Customer Support RAG Bot", layout="centered")
st.title("🤖 AI Customer Support Chatbot")
st.caption("RAG + Memory + Hard-coded URLs + Live Website Crawling")

# ============================
# ⚙️ HARD-CODED URLs HERE (edit this list!)
# ============================
DEFAULT_URLS = [
    "https://www.acronis.com/en-us/sustainability-governance/",
    "https://www.crowdstrike.com/about/environmental-social-governance/",
    "https://www.paloaltonetworks.com/about-us/corporate-responsibility",
    "https://www.fortinet.com/corporate/about-us/corporate-social-responsibility/sustainability-report",
    "https://www.veeam.com/company/corporate-governance.html",
    "https://www.commvault.com/corporate-sustainability",
    "https://www.cohesity.com/company/sustainability/",
    "https://www.zscaler.com/corporate-responsibility",
    "https://www.cyberark.com/company/esg/",
    "https://www.trendmicro.com/en_us/about/sustainability.html",
    "https://www.microsoft.com/en-us/corporate-responsibility/reports-hub",
    "https://www.cisco.com/c/en/us/about/csr/esg-hub.html",
    "https://investors.broadcom.com/esg",
    "https://www.dell.com/en-us/lp/dt/reports-and-resources",
    "https://www.ibm.com/impact",
    # Add as many as you want → they will always be indexed automatically
]

# LLM (free & blazing fast)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@st.cache_resource
def get_retriever(urls_to_crawl):
    docs = []

    # 1. Local files (optional)
    if os.path.exists("data"):
        for file in os.listdir("data"):
            path = os.path.join("data", file)
            ext = os.path.splitext(file)[1].lower()
            if ext == ".pdf":
                docs.extend(PyPDFDirectoryLoader("data").load())
            elif ext == ".txt":
                docs.extend(TextLoader(path, encoding="utf-8").load())
            elif ext == ".docx":
                docs.extend(Docx2txtLoader(path).load())

    # 2. Crawl URLs (hard-coded + user-added)
    if urls_to_crawl:
        with st.spinner(f"Crawling {len(urls_to_crawl)} URL(s)..."):
            loader = UnstructuredURLLoader(urls=urls_to_crawl, mode="single", strategy="fast")
            docs.extend(loader.load())

    if not docs:
        return None

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectorstore = Chroma.from_documents(splits, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    return vectorstore.as_retriever(search_kwargs={"k": 6})

# ============================
# Sidebar – Extra URLs
# ============================
with st.sidebar:
    st.header("Extra data sources")

    # Show hard-coded ones
    st.subheader("Always indexed (hard-coded)")
    for url in DEFAULT_URLS:
        st.write(f"✓ {url}")

    # Let user add more
    st.subheader("Add extra URLs")
    extra_input = st.text_area(
        "One URL per line (optional)",
        placeholder="https://www.acronis.com/en/sustainability-governance/",
        height=120
    )
    extra_urls = [u.strip() for u in extra_input.split("\n") if u.strip()]

    all_urls = DEFAULT_URLS + extra_urls

    if st.button("Re-index Everything", type="primary"):
        with st.spinner("Indexing all URLs + files..."):
            st.session_state.retriever = get_retriever(all_urls)
        st.success(f"Indexed {len(all_urls)} URL(s) + local files!")

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# ============================
# Initialize retriever (with hard-coded URLs from first load)
# ============================
if "retriever" not in st.session_state:
    with st.spinner("First-time indexing of hard-coded URLs..."):
        st.session_state.retriever = get_retriever(DEFAULT_URLS)

retriever = st.session_state.retriever
if not retriever:
    st.stop()

# ============================
# Chat Interface
# ============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def format_docs(docs):
    return "\n\n".join(f"**Source:** {d.metadata.get('source', 'Unknown')}\n{d.page_content}" for d in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert customer sustinability agent. Use ONLY the provided context. Always cite sources at the end of your answer."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda x: st.session_state.chat_history}
    | prompt
    | llm
    | StrOutputParser()
)

# Display chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("human").write(msg.content)
    else:
        st.chat_message("ai").write(msg.content)

# Input box – now 100% guaranteed to appear
if question := st.chat_input("Ask anything about cybesecurity peers ESG reports..."):
    st.chat_message("human").write(question)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = chain.invoke(question)
        st.write(response)

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=response))