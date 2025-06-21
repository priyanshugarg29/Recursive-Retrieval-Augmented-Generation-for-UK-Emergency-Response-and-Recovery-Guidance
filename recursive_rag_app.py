import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import json

# Streamlit app config
st.set_page_config(page_title="UK Emergency Planning Assistant", layout="wide")

# Basic description
st.title("UK Emergency Response and Recovery Guidance Assistant")
st.markdown("This tool leverages recursive retrieval-augmented generation (RAG) to assist users in exploring complex multi-part questions about the UK Government's emergency response and recovery guidance.")

# User input interface
st.subheader("Enter your question")
query = st.text_area("Type your complex policy or operational query below:", height=150)

# Submission button
submit = st.button("Submit")
reset = st.button("Reset")
download_slot = st.empty()

if reset:
    st.experimental_rerun()

# Load FAISS retriever
@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local("vector_store/emergency_guidance_index", embedding_model, allow_dangerous_deserialization=True).as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever = load_retriever()

# Function to call Gemini API
def call_gemini(prompt):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    api_key = "AIzaSyBe5J5cFtT9Uvd9sdRW1B2x3bHSK5NlIVY"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro:generateContent?key={api_key}"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"[ERROR {response.status_code}] {response.text}"

# Recursive RAG function
def recursive_rag(user_query):
    # Decompose user query using Gemini
    decomposition_prompt = f"Decompose the following query into atomic subquestions:\n\n{user_query}\n\nOutput as a numbered list."
    subq_text = call_gemini(decomposition_prompt)

    # Parse subquestions
    subquestions = [line.strip()[3:] for line in subq_text.strip().splitlines() if line.strip() and line.strip()[0].isdigit()]

    results = []
    for q in subquestions:
        docs = retriever.get_relevant_documents(q)
        context = "\n".join([doc.page_content for doc in docs])
        answer_prompt = f"Use the context to answer the question. Be precise and only answer from the content.\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:"
        ans = call_gemini(answer_prompt)
        results.append(f"Q: {q}\nA: {ans}\n")

    # Combine all sub-answers
    synthesis_prompt = f"Based on the following Q&A pairs, synthesize a complete and coherent final answer:\n\n{chr(10).join(results)}"
    final_response = call_gemini(synthesis_prompt)
    return final_response, results

# Execution on submit
if submit and query.strip():
    with st.spinner("Running recursive retrieval and generation..."):
        final_answer, breakdown = recursive_rag(query)
        st.subheader("Final Synthesized Answer")
        st.write(final_answer)
        st.subheader("Subquestion Breakdown")
        for block in breakdown:
            st.text(block)
        response_text = f"Original Query:\n{query}\n\nFinal Answer:\n{final_answer}\n\nSubquestions:\n{chr(10).join(breakdown)}"
        download_slot.download_button("Download Response", data=response_text, file_name="emergency_guidance_response.txt", mime="text/plain")