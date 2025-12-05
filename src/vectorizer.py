#Creates chunks of raw text to create semantic search vectors required for FAISS retriever and bm25 retriever
import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
vectorbase_path = "vectorstore"
embedding_model="sentence-transformers/all-MiniLM-L6-v2"
#embedding_model="all-mpnet-base-v2"
data_path = "data"
raw_chunks_path = os.path.join(vectorbase_path, "raw_chunks.pkl")
def create_chunks_and_vectorize():
    if not os.path.exists(vectorbase_path):
        os.makedirs(vectorbase_path)
        print(f"Created directory: {vectorbase_path}")
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    splitter=RecursiveCharacterTextSplitter(chunk_size=1050,chunk_overlap=250)
    chunks=splitter.split_documents(documents)

    with open(raw_chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
        print("Raw text chunks file created successfully")
    
    embedder=HuggingFaceEmbeddings(model_name=embedding_model ,model_kwargs={'device': 'cpu'})
    vectorstore=FAISS.from_documents(chunks,embedder)
    if not os.path.exists(vectorbase_path):
        os.makedirs(vectorbase_path)
    vectorstore.save_local(vectorbase_path)
    print("Vectorstore created and saved successfully")

if __name__ == "__main__":
    create_chunks_and_vectorize()

