import os
import pickle
from dotenv import load_dotenv
from langchain.community.retrievers import bm25_retriever
from langchain.community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.community.retrievers import Ensemble_retriever
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
vectorstore_path="vectorstore"
chunks_path="vectorstore/raw_chunks.pkl"
embedding_model="sentence-transformers/all-MiniLM-L6-v2"
llm_model="gpt-3.5-turbo"
load_dotenv()
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
contain any irrelevant information.The context for the query is given below:
{context}'''

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
    db_object=FAISS.load_local(vectorstore_path, embeddings)

    semantic_retriever = db_object.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    try:
        keyword_retriever = bm25_retriever.from_documents(raw_chunks, top_k=4)
    except Exception as e:
        # Fallback: try common class name
        try:
            from langchain.community.retrievers import BM25Retriever
            keyword_retriever = BM25Retriever.from_documents(raw_chunks, top_k=4)
        except Exception:
            raise RuntimeError("Could not create BM25 retriever automatically. Check langchain-community version.") from e
    try:
        ensemble_cls = Ensemble_retriever.EnsembleRetriever
    except Exception:
        # fallback try direct name
        try:
            from langchain.community.retrievers import EnsembleRetriever
            ensemble_cls = EnsembleRetriever
        except Exception:
            raise RuntimeError("Could not import EnsembleRetriever. Check your langchain-community package.")

    hybrid_retriever = ensemble_cls(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.8, 0.2]
    )
    print("Hybrid retriever created successfully.")

    llm = ChatOpenAI(model_name=llm_model, temperature=0.3)
    condense_query_chain=ChatPromptTemplate.from_messages([
        ("system",'''Given the following conversation and a follow-up question, rephrase the follow-up 
         question to be a standalone question which might require context from the conversation.
         The formulated question must be clear and concise with least number of words possible while 
         maintaining the original meaning.Do not add any additional information and don't answer the question,just
         reformulate it.If not possible,return the same'''),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{question}")])
    history_aware_retriever=create_history_aware_retriever(
        base_retriever=hybrid_retriever,condense_question_chain=condense_query_chain,llm=llm)
    answer_prompt_final=ChatPromptTemplate.from_messages([
        ("system",SystemPrompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{question}")
    ])
    doc_chain=create_stuff_documents_chain(llm=llm,prompt=answer_prompt_final)
    rag_chain=create_retrieval_chain(history_aware_retriever,doc_chain)
    print("RAG pipeline created successfully.")
    return {
        "rag_chain": rag_chain,
        "history_aware_retriever": history_aware_retriever,
        "hybrid_retriever": hybrid_retriever,
        "db_object": db_object,
        "embeddings": embeddings
    }

if __name__ == "__main__":
    pipelined_object=rag_loader_and_pipeliner()



