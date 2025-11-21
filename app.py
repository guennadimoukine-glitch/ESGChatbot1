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

st.set_page_config(page_title="ESG AI Bot", layout="centered")
st.title("ESG AI Bot Chatbot")
st.caption("Search ESG reports")

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
    # Add as many as you want → they will always be indexed automatically
]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

@st.cache_resource
def get_retriever(urls_to_crawl):
    docs = []

    # Local files (if any)
    if os.path.exists("data"):
        for file in os.listdir("data"):
            path = os.path.join("data", file)
            ext = os.path.splitext(file)[1].lower()
            if ext == ".pdf":
                docs.extend(PyPDFDirectoryLoader("data").load())
            elif ext == ".txt":
                docs.extend(TextLoader(path).load())
            elif ext == ".docx":
                docs.extend(Docx2txtLoader(path).load())

    # URLs
    if urls_to_crawl:
        with st.spinner(f"Crawling {len(urls_to_crawl)} URL(s)..."):
            loader = UnstructuredURLLoader(urls=urls_to_crawl, mode="single")
            docs.extend(loader.load())

    if not docs:
        st.info("No documents found. Add URLs and click Re-index.")
        return None

    splits = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(docs)
    vectorstore = Chroma.from_documents(splits, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    return vectorstore.as_retriever(search_kwargs={"k": 9999})  # basically everything

# Sidebar
with st.sidebar:
    st.header("Knowledge Base")
    st.subheader("Always indexed")
    for url in DEFAULT_URLS:
        st.write(f"✓ {url}")

    st.subheader("Add extra URLs (optional)")
    extra = st.text_area("One per line", height=120)
    extra_urls = [u.strip() for u in extra.split("\n") if u.strip()]

    all_urls = DEFAULT_URLS + extra_urls

    if st.button("Re-index Everything", type="primary"):
        st.session_state.retriever = get_retriever(all_urls)
        st.success(f"Successfully indexed {len(all_urls)} URL(s)!")
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# First load
if "retriever" not in st.session_state:
    with st.spinner("Indexing your default URLs…"):
        st.session_state.retriever = get_retriever(DEFAULT_URLS)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

retriever = st.session_state.retriever

# Safety check
if not retriever:
    st.stop()

# ============================
# FIXED PROMPT — this is the real fix
# ============================
def format_docs(docs):
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n".join(
        f"Source → {doc.metadata.get('source', 'Unknown')}\n{doc.page_content.strip()}"
        for doc in docs
    )

system_prompt = """You are a student researching systainabilty activities of different organizations.
Answer the question using the provided context. 
If the answer appears anywhere in the context (even partially or indirectly), use it.
Only say "I don't have that information" if it is truly not present at all in the context.

Context:
{context}

Always end your answer with the sources used."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# Pull chat history once (fixes the previous AttributeError)
current_history = st.session_state.chat_history

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda x: current_history}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("human").write(msg.content)
    else:
        st.chat_message("ai").write(msg.content)

if question := st.chat_input("Ask anything…"):
    st.chat_message("human").write(question)

    with st.chat_message("ai"):
        with st.spinner("Searching knowledge base…"):
            response = chain.invoke(question)
        st.markdown(response)

    # Append after successful response
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=response))