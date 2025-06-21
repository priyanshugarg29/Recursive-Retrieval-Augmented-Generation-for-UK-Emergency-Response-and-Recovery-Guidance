import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import json

# Page configuration
st.set_page_config(page_title="UK Emergency Planning Assistant", layout="wide")

# Background color theme-independent
st.markdown("""
    <style>
        .stApp {
            background-color: #e9f2f9;
            color: black;
        }
        textarea, .stTextInput > div > div > input {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and context
st.title("UK Emergency Response and Recovery Guidance Assistant")
st.markdown("This tool uses Retrieval-Augmented Generation (RAG) to answer complex questions using UK Government emergency planning documents.")
st.markdown("**Disclaimer:** This is an exploratory academic tool. Always follow the latest official UK Government guidance. Try an example query or enter your own.")

# Example queries
examples = [
    "What roles do Strategic Coordinating Groups and Recovery Coordinating Groups play during a crisis?",
    "How should public communications be coordinated during a major incident?",
    "What are the core responsibilities of the Local Resilience Forum?",
    "How is the transition from response to recovery handled?",
    "What contingency measures are recommended for local authorities?"
]

selected = st.selectbox("Example queries:", [""] + examples)
query = st.text_area("Enter your query:", value=selected if selected else "", height=120)

col1, col2, col3 = st.columns([1, 1, 1])
submit = col1.button("Submit")
reset = col2.button("Reset")
download = col3.empty()

if reset:
    st.experimental_rerun()

# Load vectorstore retriever
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(
        "emergency_guidance_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    ).as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever = load_vectorstore()

# Gemini-based QA
def query_with_gemini(user_query):
    retrieved_docs = retriever.get_relevant_documents(user_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""You are an emergency planning assistant. Use the provided documents to answer the user's question as accurately and clearly as possible. 
If the answer is not present, respond that the information is unavailable based on the current context.

Context:
{context}

Question:
{user_query}

Answer:
"""
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"[ERROR {response.status_code}] {response.text}"

# Handle query
if submit and query.strip():
    with st.spinner("Generating answer..."):
        response = query_with_gemini(query)
        st.subheader("Answer")
        st.write(response)
        result_text = f"Query:\n{query}\n\nAnswer:\n{response}"
        download.download_button(
            label="Download Answer",
            data=result_text,
            file_name="emergency_guidance_response.txt",
            mime="text/plain"
        )