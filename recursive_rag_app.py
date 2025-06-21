import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import json

# Streamlit app config
st.set_page_config(page_title="UK Emergency Planning Assistant", layout="wide")

# Apply custom CSS for visibility in all themes
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f7fa;
            color: black;
        }
        textarea, .stTextInput > div > div > input {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("UK Emergency Response and Recovery Guidance Assistant")
st.markdown("This tool uses recursive retrieval-augmented generation (RAG) to help explore complex questions based on the UK's emergency response and recovery guidance.")
st.markdown("**Disclaimer:** This project is exploratory. Always refer to the latest UK Government guidance when making decisions regarding emergency response and planning.")

# Example prompts
examples = [
    "What roles do Strategic Coordinating Groups and Recovery Coordinating Groups play during a crisis?",
    "How should public communications be coordinated during a major incident?",
    "What are the core responsibilities of the Local Resilience Forum?",
    "How is the transition from response to recovery handled?",
    "What contingency measures are recommended for local authorities?"
]

# Input box with example query selection
selected_example = st.selectbox("Example queries:", [""] + examples)
query = st.text_area("Enter your query:", value=selected_example if selected_example else "", height=150)

# Submission button
submit = st.button("Submit")
download_slot = st.empty()

# Load FAISS retriever
@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(
    "emergency_guidance_index",  # correct path relative to repo root
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
).as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever = load_retriever()

# Function to call Gemini API
def call_gemini(prompt):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro:generateContent?key={api_key}"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"[ERROR {response.status_code}] {response.text}"

# Recursive RAG function
def recursive_rag(user_query):
    decomposition_prompt = f"Decompose the following query into atomic subquestions:\n\n{user_query}\n\nOutput as a numbered list."
    subq_text = call_gemini(decomposition_prompt)
    subquestions = [line.strip()[3:] for line in subq_text.strip().splitlines() if line.strip() and line.strip()[0].isdigit()]
    results = []
    for q in subquestions:
        docs = retriever.get_relevant_documents(q)
        context = "\n".join([doc.page_content for doc in docs])
        answer_prompt = f"Use the context to answer the question. Be precise and only answer from the content.\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:"
        ans = call_gemini(answer_prompt)
        results.append(f"Q: {q}\nA: {ans}\n")
    synthesis_prompt = f"Based on the following Q&A pairs, synthesize a complete and coherent final answer:\n\n{chr(10).join(results)}"
    final_response = call_gemini(synthesis_prompt)
    return final_response, results

# Process query
if submit and query.strip():
    with st.spinner("Running recursive retrieval and generation..."):
        final_answer, breakdown = recursive_rag(query)
        st.subheader("Final Synthesized Answer")
        st.write(final_answer)
        st.subheader("Subquestion Breakdown")
        for block in breakdown:
            st.text(block)
        result_text = f"Original Query:\n{query}\n\nFinal Answer:\n{final_answer}\n\nSubquestions:\n{chr(10).join(breakdown)}"
        download_slot.download_button("Download Response", data=result_text, file_name="emergency_guidance_response.txt", mime="text/plain")
