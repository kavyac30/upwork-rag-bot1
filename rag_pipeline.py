import os
import time
import requests
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

API_KEY = st.secrets["DEEPINFRA_API_KEY"]
MODEL_NAME = st.secrets["MODEL_NAME"]

CHROMA_PATH = "chroma_db"

# Load embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


# def retrieve_docs(query):
#     docs = retriever.invoke(query)
#     return docs

def retrieve_docs(query):

    docs = retriever.invoke(query)

    print("\n===== RETRIEVED CHUNKS =====")

    for i, doc in enumerate(docs, 1):

        print(f"\n--- Chunk {i} ---\n")

        print(doc.page_content)

    return docs

SYSTEM_PROMPT = """
You are a Senior Upwork API Consultant.

Your task is to answer ONLY from the provided documentation context.

Rules:
1. Do NOT hallucinate.
2. Do NOT use outside knowledge.
3. If answer is unavailable in context, reply EXACTLY:
"I'm sorry, but the provided documentation does not contain that information."
4. Keep answers technical and concise.
"""


def generate_response(query, docs):

    context = "\n\n".join([doc.page_content for doc in docs])

    user_prompt = f"""
Documentation Context:
{context}

User Question:
{query}
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }

    start_time = time.time()

    response = requests.post(
        "https://api.deepinfra.com/v1/openai/chat/completions",
        headers=headers,
        json=payload
    )

    end_time = time.time()

    latency = round(end_time - start_time, 2)

    # result = response.json()

    # answer = result["choices"][0]["message"]["content"]
    result = response.json()

    print(result)

    if "choices" in result:
        answer = result["choices"][0]["message"]["content"]
    else:
        answer = f"API Error: {result}"

    return answer, latency