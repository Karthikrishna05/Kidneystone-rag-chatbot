import os
import pickle
from dotenv import load_dotenv
from collections import defaultdict
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List,Any
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
vectorstore_path="vectorstore"
chunks_path="vectorstore/raw_chunks.pkl"
embedding_model="sentence-transformers/all-MiniLM-L6-v2"
llm_model="meta-llama/llama-3.3-70b-instruct:free"
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
print(f"DEBUG: Loaded API Key: {api_key[:10]}..." if api_key else "DEBUG: API Key is None!")
SystemPrompt='''You are a helpful AI assistant specialized in answering questions about kidney stones based on the provided context.
 Use the context to provide accurate and concise answers.The answer must contain only precise statements relevant 
pertaining to the query posed and shall not include the thinking process of the AI. If the context does not contain sufficient 
information to answer the question, answer with "The provided context does not contain sufficient information to answer the question".
1.Do not fabricate any information or hallucinate details to answer and make sure to keep the answer grounded to
the context provided.
2.You are not a medically certified professional, so avoid giving any medical advice.If asked for a medical opinion or advice,politely
deny by them saying "I am an AI language model and not a medically certified professional, so I cannot provide medical advice.
Please consult a qualified healthcare professional for any medical concerns."
3.If user asks any questions unrelated to kidney stones, politely inform them that your expertise is limited to kidney stones and 
you cannot provide information on other topics.
Answers must be summarized in a concise manner with all relevant points covered from the context and must not
contain any irrelevant information.Even try to cite the sources in the response.The context for the query is given below:
{context}'''
class CustomHybridRetriever(BaseRetriever):
    """
    This custom retriever implements Reciprocal Rank Fusion (RRF)
    to combine and re-rank results from BM25 (keyword) and FAISS (semantic).
    """
    faiss_retriever: Any 
    bm25_retriever: Any   
    k: int = 5
    rrf_k: int = 60

    def get_relevant_documents(self, query: str) -> List[Document]:
       
        self.faiss_retriever.search_kwargs = {"k": self.k}
        self.bm25_retriever.k = self.k

        faiss_docs = self.faiss_retriever.get_relevant_documents(query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        
    
        rrf_scores = defaultdict(float)
        doc_store = {}
      
        for i, doc in enumerate(faiss_docs):
            content = doc.page_content
            if content not in doc_store:
                doc_store[content] = doc
            rrf_scores[content] += 1.0 / (self.rrf_k + i + 1)

        for i, doc in enumerate(bm25_docs):
            content = doc.page_content
            if content not in doc_store:
                doc_store[content] = doc
            rrf_scores[content] += 1.0 / (self.rrf_k + i + 1)
        
        sorted_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_docs = [doc_store[content] for content in sorted_contents]
        
        return final_docs[:self.k]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # Asynchronous version (required by the base class)
        return self.get_relevant_documents(query)
    

def rag_loader_and_pipeliner():
    print("Loading raw chunks from vectorstore...")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run the vectorizer first.")
    with open(chunks_path, "rb") as f:
        raw_chunks = pickle.load(f)

    print("Setting up embeddings and retrievers...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model ,model_kwargs={'device': 'cpu'})
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vectorstore directory not found at {vectorstore_path}. Run the vectorizer first.")
    db_object=FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    faiss_retriever = db_object.as_retriever()
    keyword_retriever = BM25Retriever.from_documents(raw_chunks)
   
    hybrid_retriever = CustomHybridRetriever(
        faiss_retriever=faiss_retriever,
        bm25_retriever=keyword_retriever,
        k=5
    )
    print("Hybrid retriever created successfully.")

    llm = ChatOpenAI(model_name=llm_model, temperature=0.3,openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1", # Point to OpenRouter
        default_headers={
            "HTTP-Referer": "http://localhost:8501", # Required by OpenRouter
            "X-Title": "KidneyStoneChatbot"          # Required by OpenRouter
        })
    condense_query_chain=ChatPromptTemplate.from_messages([
        ("system",'''Given the following conversation and a follow-up question, rephrase the follow-up 
         question to be a standalone question which might require context from the conversation.
         The formulated question must be clear and concise with least number of words possible while 
         maintaining the original meaning.Do not add any additional information and don't answer the question,just
         reformulate it.If not possible,return the same'''),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")])
    history_aware_retriever=create_history_aware_retriever(
        llm,hybrid_retriever,condense_query_chain)
    answer_prompt_final=ChatPromptTemplate.from_messages([
        ("system",SystemPrompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ])
    doc_chain=create_stuff_documents_chain(llm=llm,prompt=answer_prompt_final)
    rag_chain=create_retrieval_chain(history_aware_retriever,doc_chain)
    print("RAG pipeline created successfully.")
    return rag_chain


