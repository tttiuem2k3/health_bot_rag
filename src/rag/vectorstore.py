from typing import Union, List
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pickle
import torch

class VectorDB:
    def __init__(self,
                 documents: List[Document] = None,
                 vector_db: Union[Chroma, FAISS] = Chroma,
                 embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder",
                 load_db_path ="No"
                 ) -> None:
        
        self.vector_db = vector_db
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        if load_db_path == "No":
            self.db = self._build_db(documents)
        else:
            self.db = self._load_embedded_data(load_db_path, embedding_model)

    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        
        processed_documents = []
        for doc in documents:
            tokenized_content = " ".join(doc.page_content.split())  # Loại bỏ khoảng trắng thừa
            processed_doc = Document(page_content=tokenized_content, metadata=doc.metadata)
            processed_documents.append(processed_doc)
        return processed_documents

    def _build_db(self, documents: List[Document]):
        """
        Xây dựng cơ sở dữ liệu vector từ danh sách tài liệu.
        :param documents: Danh sách các tài liệu.
        :return: Cơ sở dữ liệu vector.
        """
        print(f"🧠 Đang Embedding dữ liệu Data...")
        if documents:
            # Tiền xử lý tài liệu trước khi tạo embedding
            documents = self._preprocess_documents(documents)
        db = self.vector_db.from_documents(documents=documents, 
                                           embedding=self.embedding)
        print("✅ Dữ liệu đã được nạp vào Chroma Database.")
        return db
    
    def _load_embedded_data(self, persist_directory: str, embedding_model: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": device})
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        print("✅ Đã load dữ liệu embedding từ Chroma Database.")
        return vectordb
        
    
    def get_retriever(self, 
                      search_type: str = "similarity", 
                      search_kwargs: dict = {"k": 5}
                      ):
        """
        Tạo retriever để tìm kiếm tài liệu dựa trên độ tương đồng vector.
        :param search_type: Loại tìm kiếm (mặc định là "similarity").
        :param search_kwargs: Tham số tìm kiếm (ví dụ: số lượng kết quả "k").
        :return: Retriever.
        """
        retriever = self.db.as_retriever(search_type=search_type,
                                         search_kwargs=search_kwargs)
        return retriever