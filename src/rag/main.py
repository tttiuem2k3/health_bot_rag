from pydantic import BaseModel, Field
from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG
from src import config
from pyvi import ViTokenizer


def preprocess_query(query: str) -> str:
    return ViTokenizer.tokenize(query.strip())

def load_disease_list(disease_file):
    with open(disease_file, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f.readlines() if line.strip()]

def detect_disease_in_query(query: str, disease_list: list):
    query_lower = query.lower()
    sorted_disease_list = sorted(disease_list, key=len, reverse=True)
    for disease in sorted_disease_list:
        if disease in query_lower:
            return disease
    return None

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")


def build_rag_chain(llm, data_dir, data_type, data_embedded="No", disease_list_file=None):
    if data_embedded == "Yes":
        vector_db = VectorDB(documents=None, load_db_path=config.Data_health_embedding_path).db
        disease_list = load_disease_list(disease_list_file)

        def custom_retriever(question: str):
            detected_disease = detect_disease_in_query(question, disease_list)
            query = preprocess_query(question)
            if detected_disease:
                print(f"🔍 Phát hiện tên bệnh trong câu hỏi: {detected_disease.title()}")
                retriever_with_filter = vector_db.as_retriever(
                    search_kwargs={"k": 7, "filter": {"title": detected_disease}}
                )
                # return retriever_with_filter.get_relevant_documents(query)
                return retriever_with_filter.invoke(query)
            else:
                retriever = vector_db.as_retriever(search_kwargs={"k": 7})
                # return retriever_with_filter.get_relevant_documents(query)
                return retriever.invoke(query)
        # retriever = vector_db.as_retriever(search_kwargs={"k": 7})

        rag_chain = Offline_RAG(llm).get_chain(custom_retriever)
    else:
        doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
        retriever = VectorDB(documents=doc_loaded).get_retriever()
        rag_chain = Offline_RAG(llm).get_chain(retriever)

    return rag_chain


