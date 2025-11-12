#Creates chunks of raw text to create semantic search vectors required for FAISS retriever and bm25 retriever
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.community.document_loaders import PyPDFDirectoryLoader
from langchain.community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
vectorbase_path = "vectorstore"
embedding_model="sentence-transformers/all-MiniLM-L6-v2"
#embedding_model="all-mpnet-base-v2"
data_path = "data"
raw_chunks_path = r"C:\Users\pc\Project-Programming\Python\Kidneystone-rag-chatbot\vectorstore\raw_chunks.pkl"
def create_chunks_and_vectorize():
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

