from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    rag_search = RAGSearch()
    query = "What is Blockchain?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)