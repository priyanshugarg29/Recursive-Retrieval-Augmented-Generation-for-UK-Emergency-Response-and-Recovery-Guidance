import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import json

# Set Streamlit page configuration
st.set_page_config(page_title="Credit Risk Policy Assistant", layout="wide")

# Apply CSS for theme-independent visibility
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f7fa;
        }
        textarea, .stTextInput > div > div > input, .css-qrbaxs, .css-1cpxqw2 {
            color: black !important;
        }
        .css-1rs6os.edgvbvh3 {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📄 Credit Risk Policy Assistant (RAG-powered)")
st.markdown("""
Welcome to the Credit Risk Assistant powered by Retrieval-Augmented Generation (RAG).
This tool allows you to query synthetic internal credit policies inspired by UK FCA CONC guidance.
""")
st.markdown("**Try one of the example prompts below or ask your own question!**")

# Example prompts
examples = [
    "What is the minimum income required for a personal loan?",
    "Are startups eligible for SME loans?",
    "Can a customer with a high-performance car apply for auto finance?",
    "Who can approve loans above £25,000?",
    "What are the risk flags for personal loan applicants?"
]

selected = st.selectbox("🧪 Example queries:", [""] + examples)
query = st.text_area("💬 Enter your query:", value=selected if selected else "", height=120)

col1, col2, col3 = st.columns([1, 1, 1])
submit = col1.button("🚀 Submit")
reset = col2.button("♻️ Reset")
download = col3.empty()

if reset:
    st.experimental_rerun()

# Load FAISS retriever
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local("faiss_index", embedding_model).as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever = load_vectorstore()

# Gemini API call
def query_with_gemini(user_query):
    retrieved_docs = retriever.get_relevant_documents(user_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""You are a credit policy assistant. Use the provided policy documents to answer the user's question.
If the answer is not available in the documents, say so clearly.

Context:
{context}

Question:
{user_query}

Helpful Answer:
"""
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

# Process query
if submit and query.strip():
    with st.spinner("🔍 Generating answer..."):
        response = query_with_gemini(query)
        st.subheader("🧠 Answer")
        st.write(response)
        result_text = f"Query:\n{query}\n\nAnswer:\n{response}"
        download.download_button(
            label="📥 Download Answer",
            data=result_text,
            file_name="credit_policy_response.txt",
            mime="text/plain"
        )