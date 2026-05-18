import os
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_PATH = "data/API Documentation Partial.pdf"
CHROMA_PATH = "chroma_db"

# Load PDF
# loader = PyPDFLoader(PDF_PATH)
loader = PDFPlumberLoader(PDF_PATH)
documents = loader.load()


# Combine text
full_text = "\n".join([doc.page_content for doc in documents])

# Sanity Check
print("\n===== SANITY CHECK =====")
print(f"Total Characters: {len(full_text)}")
print("\nSample Text:\n")
print(full_text[:1000])

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=250
)

chunks = text_splitter.split_documents(documents)

print(f"\nTotal Chunks Created: {len(chunks)}")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)

vectorstore.persist()

print("\nVector Database Created Successfully!")